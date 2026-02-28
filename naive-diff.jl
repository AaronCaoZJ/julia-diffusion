using Flux, Zygote, CUDA, cuDNN, Random, Statistics, CairoMakie, Printf

# DDPM: VP-diffusion on 2D Two-Moons
# q(x_t|x_0) = N(sqrt(ᾱ_t)·x_0, (1-ᾱ_t)·I),  reverse: predict ε

const DEVICE     = CUDA.functional() ? gpu : cpu
const SAVE_DIR   = "output/navie-out"
const T          = 1000
const BETA_START = 1e-4
const BETA_END   = 0.02
const betas      = Float32.(range(BETA_START, BETA_END, length=T))
const alphas     = 1f0 .- betas
const alpha_bar  = cumprod(alphas)   # ᾱ_t = ∏_{s=1}^{t} αs

# --- Data ---

function make_moons(n::Int; noise::Float32=0.05f0)
    n_half = n ÷ 2
    θ1   = Float32.(range(0, π, length=n_half))
    θ2   = Float32.(range(0, π, length=n - n_half))
    x1   = hcat(cos.(θ1), sin.(θ1))
    x2   = hcat(1f0 .- cos.(θ2), 1f0 .- sin.(θ2) .- 0.5f0)
    data = vcat(x1, x2)
    data .+= noise .* randn(Float32, size(data))
    return data
end

# --- Forward ---

# q(x_t | x_0): x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε
function q_sample(x0::AbstractMatrix, t::AbstractVector)
    ab         = alpha_bar[t]
    sqr_ab     = reshape(sqrt.(ab),        :, 1) |> DEVICE
    sqr_one_ab = reshape(sqrt.(1f0 .- ab), :, 1) |> DEVICE
    ε   = randn!(similar(x0))
    x_t = sqr_ab .* x0 .+ sqr_one_ab .* ε
    return x_t, ε
end

# --- Model ---

# Time-conditioned MLP: (x, t/T) -> ε
function build_model(data_dim=2, hidden=128)
    return Chain(
        Dense(data_dim + 1, hidden, gelu),
        Dense(hidden, hidden, gelu),
        Dense(hidden, hidden, gelu),
        Dense(hidden, data_dim),
    )
end

t_embed(t::AbstractVector) = reshape(Float32.(t) ./ Float32(T), :, 1) |> DEVICE

function model_forward(model, x_t, t)
    te  = t_embed(t)
    inp = hcat(x_t, te)
    return model(inp')'
end

# --- Reverse ---

function ddpm_sample(model, n_samples::Int; n_steps=T)
    x = randn(Float32, n_samples, 2) |> DEVICE
    for i in n_steps:-1:1
        t_vec  = fill(i, n_samples)
        ε_pred = model_forward(model, x, t_vec)
        β      = betas[i]
        α      = alphas[i]
        ab     = alpha_bar[i]
        coef   = β / sqrt(1f0 - ab)
        x_mean = (x .- coef .* ε_pred) ./ sqrt(α)
        if i > 1
            x = x_mean .+ sqrt(β) .* randn!(similar(x))
        else
            x = x_mean
        end
    end
    return x |> cpu
end

# --- Training ---

function warmup!(model, opt_state)
    print("JIT warmup... ")
    t_w    = time()
    x0     = make_moons(32) |> DEVICE
    t_     = rand(1:T, 32)
    x_t, ε = q_sample(x0, t_)
    _, grads = Flux.withgradient(model) do m
        ε_pred = model_forward(m, x_t, t_)
        mean((ε_pred .- ε).^2)
    end
    Flux.update!(opt_state, model, grads[1])
    CUDA.functional() && CUDA.synchronize()
    @printf("done (%.1fs)\n", time() - t_w)
end

function train_diffusion(; epochs=5000, batch_size=2048, lr=3e-3, log_interval=500)
    model     = build_model() |> DEVICE
    opt_state = Flux.setup(Adam(lr), model)
    losses    = Float32[]

    warmup!(model, opt_state)

    t_total_start  = time()
    t_window_start = time()

    for epoch in 1:epochs
        data_cpu = make_moons(batch_size)
        t        = rand(1:T, batch_size)
        data     = data_cpu |> DEVICE
        x_t, ε   = q_sample(data, t)

        loss, grads = Flux.withgradient(model) do m
            ε_pred = model_forward(m, x_t, t)
            mean((ε_pred .- ε).^2)
        end

        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)

        if epoch % log_interval == 0
            t_now      = time()
            elapsed    = t_now - t_total_start
            window     = t_now - t_window_start
            iter_per_s = log_interval / window
            eta        = (epochs - epoch) / iter_per_s
            @printf("Epoch %5d / %d  loss = %.5f  |  %.1f iter/s  elapsed = %.1fs  ETA = %.1fs\n",
                    epoch, epochs, loss, iter_per_s, elapsed, eta)
            t_window_start = t_now
        end

        if epoch % 1000 == 0 || epoch == epochs
            visualize(model, data_cpu, epoch)
        end
    end

    total = time() - t_total_start
    @printf("\nTraining done: %.2fs total, %.1f iter/s average\n", total, epochs / total)
    return model, losses
end

# --- Visualization ---

function visualize(model, x_gt, epoch)
    x_gen = ddpm_sample(model, 2000; n_steps=T)
    fig = Figure(size=(900, 400))
    ax1 = Axis(fig[1, 1], title="Ground Truth",
               limits=((-1.5, 2.5), (-1.5, 2.0)))
    scatter!(ax1, x_gt[1:2000, 1], x_gt[1:2000, 2];
             markersize=3, alpha=0.6, color=:steelblue)
    ax2 = Axis(fig[1, 2], title="DDPM (epoch=$epoch)",
               limits=((-1.5, 2.5), (-1.5, 2.0)))
    scatter!(ax2, x_gen[:, 1], x_gen[:, 2];
             markersize=3, alpha=0.6, color=:tomato)
    path = joinpath(SAVE_DIR, "jl_epoch_$(lpad(epoch, 5, '0')).png")
    save(path, fig)
    println("Saved: $path")
end

function plot_loss(losses)
    fig = Figure()
    ax  = Axis(fig[1, 1], xlabel="Epoch", ylabel="MSE Loss",
               title="Training Loss", yscale=log10)
    lines!(ax, losses, color=:black)
    path = joinpath(SAVE_DIR, "jl_loss.png")
    save(path, fig)
    println("Saved: $path")
end

# --- Main ---

println("Using device: " * (CUDA.functional() ? "CUDA GPU" : "CPU"))
mkpath(SAVE_DIR)
println("Training DDPM (2D Two Moons)...")
model, losses = train_diffusion(epochs=5000, batch_size=2048, lr=3e-3)
println("Training done!")
plot_loss(losses)
