using Flux, Zygote, CUDA, cuDNN, Random, Statistics, CairoMakie, Printf

# VP-SDE (Variance Preserving SDE) -- Score-based Diffusion
# Song et al. "Score-Based Generative Modeling through SDEs" (2021)
#
# Forward SDE:  dx = -β(t)/2 · x dt + √β(t) dW,  t ∈ [0,1]
#
# Perturbation kernel:
#   p(x_t | x_0) = N(α(t)·x_0, σ²(t)·I)
#   α(t) = exp(-1/2 ∫₀ᵗ β(s) ds),  σ(t) = √(1 - α²(t))
#
# Score:    ∇_x log p_t(x_t|x_0) = -ε / σ(t)
# Training: ε_θ(x_t, t) ≈ ε  (MSE noise prediction)
#
# Reverse SDE (Euler-Maruyama, t: 1->0):
#   Δx = [β/2·x + β·s_θ]·Δt + √(β·Δt)·z,  z ~ N(0,I)

const DEVICE      = CUDA.functional() ? gpu : cpu
const SEED        = 42
const SAVE_DIR    = "output/sde-out"
const T_EMBED_DIM = 8
const FREQS       = (2f0 .^ Float32.(0:T_EMBED_DIM÷2-1)) |> DEVICE
const BETA_MIN    = 0.1f0
const BETA_MAX    = 20.0f0
const T_EPS       = 1f-3
const N_STEPS     = 200

beta_fn(t)     = BETA_MIN + t * (BETA_MAX - BETA_MIN)
int_beta_fn(t) = BETA_MIN * t + 0.5f0 * (BETA_MAX - BETA_MIN) * t^2
alpha_fn(t)    = exp(-0.5f0 * int_beta_fn(t))
sigma_fn(t)    = sqrt(max(1f0 - alpha_fn(t)^2, 1f-8))

# --- Data ---

function cat_shape(t)
    x = -(721*sin(t))/4+196/3*sin(2*t)-86/3*sin(3*t)-131/2*sin(4*t)+477/14*sin(5*t)+27*sin(6*t)-29/2*sin(7*t)+68/5*sin(8*t)+1/10*sin(9*t)+23/4*sin(10*t)-19/2*sin(12*t)-85/21*sin(13*t)+2/3*sin(14*t)+27/5*sin(15*t)+7/4*sin(16*t)+17/9*sin(17*t)-4*sin(18*t)-1/2*sin(19*t)+1/6*sin(20*t)+6/7*sin(21*t)-1/8*sin(22*t)+1/3*sin(23*t)+3/2*sin(24*t)+13/5*sin(25*t)+sin(26*t)-2*sin(27*t)+3/5*sin(28*t)-1/5*sin(29*t)+1/5*sin(30*t)+(2337*cos(t))/8-43/5*cos(2*t)+322/5*cos(3*t)-117/5*cos(4*t)-26/5*cos(5*t)-23/3*cos(6*t)+143/4*cos(7*t)-11/4*cos(8*t)-31/3*cos(9*t)-13/4*cos(10*t)-9/2*cos(11*t)+41/20*cos(12*t)+8*cos(13*t)+2/3*cos(14*t)+6*cos(15*t)+17/4*cos(16*t)-3/2*cos(17*t)-29/10*cos(18*t)+11/6*cos(19*t)+12/5*cos(20*t)+3/2*cos(21*t)+11/12*cos(22*t)-4/5*cos(23*t)+cos(24*t)+17/8*cos(25*t)-7/2*cos(26*t)-5/6*cos(27*t)-11/10*cos(28*t)+1/2*cos(29*t)-1/5*cos(30*t)
    y = -(637*sin(t))/2-188/5*sin(2*t)-11/7*sin(3*t)-12/5*sin(4*t)+11/3*sin(5*t)-37/4*sin(6*t)+8/3*sin(7*t)+65/6*sin(8*t)-32/5*sin(9*t)-41/4*sin(10*t)-38/3*sin(11*t)-47/8*sin(12*t)+5/4*sin(13*t)-41/7*sin(14*t)-7/3*sin(15*t)-13/7*sin(16*t)+17/4*sin(17*t)-9/4*sin(18*t)+8/9*sin(19*t)+3/5*sin(20*t)-2/5*sin(21*t)+4/3*sin(22*t)+1/3*sin(23*t)+3/5*sin(24*t)-3/5*sin(25*t)+6/5*sin(26*t)-1/5*sin(27*t)+10/9*sin(28*t)+1/3*sin(29*t)-3/4*sin(30*t)-(125*cos(t))/2-521/9*cos(2*t)-359/3*cos(3*t)+47/3*cos(4*t)-33/2*cos(5*t)-5/4*cos(6*t)+31/8*cos(7*t)+9/10*cos(8*t)-119/4*cos(9*t)-17/2*cos(10*t)+22/3*cos(11*t)+15/4*cos(12*t)-5/2*cos(13*t)+19/6*cos(14*t)+7/4*cos(15*t)+31/4*cos(16*t)-cos(17*t)+11/10*cos(18*t)-2/3*cos(19*t)+13/3*cos(20*t)-5/4*cos(21*t)+2/3*cos(22*t)+1/4*cos(23*t)+5/6*cos(24*t)+3/4*cos(26*t)-1/2*cos(27*t)-1/10*cos(28*t)-1/3*cos(29*t)-1/19*cos(30*t)
    return Float32[x, y]
end

function random_literal_cat(n::Int; sigma::Float32=0.05f0)
    pts = reduce(hcat, [cat_shape(rand() * 2π) / 200 for _ in 1:n])
    pts .+= randn(Float32, 2, n) .* sigma
    return Matrix(pts')
end

# --- Model ---

# MLP ε-predictor: input = (x_t ‖ t_embed), output = ε
function build_model(data_dim=2, hidden=256)
    return Chain(
        Dense(data_dim + T_EMBED_DIM, hidden, gelu),
        Dense(hidden, hidden, gelu),
        Dense(hidden, hidden, gelu),
        Dense(hidden, data_dim),
    )
end

function t_embed(t::AbstractVector)
    te = reshape(Float32.(t) |> DEVICE, :, 1) .* reshape(FREQS, 1, :)
    return hcat(sin.(te), cos.(te))
end

function model_forward(model, x_t, t)
    te  = t_embed(t)
    inp = hcat(x_t, te)
    return model(inp')'
end

# --- Forward ---

# p(x_t | x_0) = N(α(t)·x_0, σ²(t)·I)
function vpsde_forward(x0::AbstractMatrix, t::AbstractVector)
    at = reshape(alpha_fn.(Float32.(t)), :, 1) |> DEVICE
    st = reshape(sigma_fn.(Float32.(t)), :, 1) |> DEVICE
    ε  = randn!(similar(x0))
    return at .* x0 .+ st .* ε, ε
end

# --- Reverse ---

# Euler-Maruyama reverse SDE, t: 1->0
function sde_sample(model, n_samples::Int; n_steps=N_STEPS, verbose=false)
    x  = randn(Float32, n_samples, 2) |> DEVICE
    dt = Float32(1.0 / n_steps)

    CUDA.functional() && CUDA.synchronize()
    t0 = time()

    for i in n_steps:-1:1
        t_val = Float32(i) / Float32(n_steps)
        t_vec = fill(t_val, n_samples)
        β     = beta_fn(t_val)
        st    = sigma_fn(t_val)
        score = -model_forward(model, x, t_vec) ./ st
        drift = (β/2f0 .* x .+ β .* score) .* dt
        noise = i > 1 ? sqrt(β * dt) .* randn!(similar(x)) : zero(x)
        x     = x .+ drift .+ noise
    end

    CUDA.functional() && CUDA.synchronize()
    verbose && @printf("  SDE | steps=%4d  time=%.3fs\n", n_steps, time() - t0)
    return Array(x)
end

function sde_sample_with_traj(model, n_samples::Int; n_steps=N_STEPS, x0=nothing)
    x    = isnothing(x0) ? randn(Float32, n_samples, 2) |> DEVICE : x0 |> DEVICE
    dt   = Float32(1.0 / n_steps)
    traj = Vector{Matrix{Float32}}(undef, n_steps + 1)
    traj[1] = Array(x)

    for i in n_steps:-1:1
        t_val = Float32(i) / Float32(n_steps)
        t_vec = fill(t_val, n_samples)
        β     = beta_fn(t_val)
        st    = sigma_fn(t_val)
        score = -model_forward(model, x, t_vec) ./ st
        drift = (β/2f0 .* x .+ β .* score) .* dt
        noise = i > 1 ? sqrt(β * dt) .* randn!(similar(x)) : zero(x)
        x     = x .+ drift .+ noise
        traj[n_steps - i + 2] = Array(x)
    end

    return Array(x), traj
end

# --- Training ---

function warmup!(model, opt_state)
    print("JIT warmup... ")
    t_w = time()
    x0  = random_literal_cat(32) |> DEVICE
    t   = T_EPS .+ rand(Float32, 32) .* (1f0 - T_EPS)
    _, grads = Flux.withgradient(model) do m
        x_t, ε = vpsde_forward(x0, t)
        ε_pred = model_forward(m, x_t, t)
        mean((ε_pred .- ε).^2)
    end
    Flux.update!(opt_state, model, grads[1])
    CUDA.functional() && CUDA.synchronize()
    @printf("done (%.1fs)\n", time() - t_w)
end

function train_sde(; epochs=5000, batch_size=2048, lr=3e-3, log_interval=500)
    model     = build_model() |> DEVICE
    opt_state = Flux.setup(Adam(lr), model)

    warmup!(model, opt_state)
    println("Training ($epochs steps)...")
    t_start = time()

    for epoch in 1:epochs
        x0 = random_literal_cat(batch_size) |> DEVICE
        t  = T_EPS .+ rand(Float32, batch_size) .* (1f0 - T_EPS)

        loss, grads = Flux.withgradient(model) do m
            x_t, ε = vpsde_forward(x0, t)
            ε_pred = model_forward(m, x_t, t)
            mean((ε_pred .- ε).^2)
        end

        Flux.update!(opt_state, model, grads[1])

        if epoch % log_interval == 0
            @printf("  step %5d / %d  loss = %.5f  elapsed = %.1fs\n",
                    epoch, epochs, loss, time() - t_start)
        end
    end
    @printf("Training done: %.1fs\n", time() - t_start)
    return model
end

# --- Visualization ---

function visualize_cat(model; n_samples=3000, n_steps=N_STEPS)
    x_gen = sde_sample(model, n_samples; n_steps=n_steps)
    x_gt  = random_literal_cat(n_samples)

    n_traj  = 5
    x0_traj = rand(Float32, n_traj, 2) .+ 2.0f0
    _, traj    = sde_sample_with_traj(model, n_traj; n_steps=n_steps, x0=x0_traj)
    n_traj_pts = length(traj)

    fig = Figure(size=(800, 400))

    ax1 = Axis(fig[1, 1];
               title="Generated vs Reference (VP-SDE)",
               xlabel="x1", ylabel="x2", aspect=DataAspect(),
               limits=(-3f0, 3f0, -3f0, 3f0))
    scatter!(ax1, x_gt[:,1],  x_gt[:,2];  markersize=3, color=(:steelblue, 0.25), label="Reference")
    scatter!(ax1, x_gen[:,1], x_gen[:,2]; markersize=3, color=(:tomato,    0.5),  label="Generated")
    axislegend(ax1; position=:lt, labelsize=9)

    ax2 = Axis(fig[1, 2];
               title="Particle Trajectories (Noisy Points -> Cat)",
               xlabel="x1", ylabel="x2", aspect=DataAspect(),
               limits=(-3f0, 4f0, -3f0, 4f0))
    scatter!(ax2, x_gt[:,1], x_gt[:,2]; markersize=2, color=(:steelblue, 0.125))
    x_end  = traj[end]
    plasma = Makie.to_colormap(:plasma)
    for i in 1:n_traj
        xs = [traj[s][i, 1] for s in 1:n_traj_pts]
        ys = [traj[s][i, 2] for s in 1:n_traj_pts]
        if i == 1
            lines!(ax2, xs, ys; color=LinRange(0f0, 1f0, n_traj_pts),
                   colormap=(:plasma, 0.8), linewidth=0.8)
            scatter!(ax2, [x0_traj[i,1]], [x0_traj[i,2]];
                     markersize=9, color=plasma[1],   marker=:circle, label="Start (noise)")
            scatter!(ax2, [x_end[i,1]],   [x_end[i,2]];
                     markersize=9, color=plasma[end], marker=:circle, label="End (data)")
        else
            lines!(ax2, xs, ys; color=(:tomato, 0.15), linewidth=0.9)
            scatter!(ax2, [x0_traj[i,1]], [x0_traj[i,2]];
                     markersize=9, color=(:black, 0.15), marker=:circle)
            scatter!(ax2, [x_end[i,1]],   [x_end[i,2]];
                     markersize=9, color=(:tomato, 0.15), marker=:circle)
        end
    end
    axislegend(ax2; position=:lt, labelsize=9)

    mkpath(SAVE_DIR)
    path = joinpath(SAVE_DIR, "sde_cat.png")
    save(path, fig)
    println("Saved: $path")
end

# --- Main ---

Random.seed!(SEED)
CUDA.functional() && CUDA.seed!(SEED)
println("Using device: " * (CUDA.functional() ? "CUDA GPU" : "CPU"))
println("Starting VP-SDE Score Diffusion (2D Cat)...")
model = train_sde(epochs=8000, batch_size=2048, lr=3e-3)
visualize_cat(model)
