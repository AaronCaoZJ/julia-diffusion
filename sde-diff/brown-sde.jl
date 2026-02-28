using Flux, Zygote, CUDA, cuDNN, Random, Statistics, CairoMakie, Printf

# Brownian Bridge Flow Matching
# Time axis: t=1 (noise X_1 ~ U([2,3]²)), t=0 (data X_0)
#
# Bridge (training):
#   x_t = (1-t)·x_0 + t·x_1 + σ·√(t(1-t))·ε,  ε ~ N(0,I)
#
# Model: x̂_0 = model(t, x_t)  (predict data endpoint directly)
#
# Exact bridge step (denoising, t: 1->0):
#   x_{t2} = (t1-t2)/t1 · x̂_0 + t2/t1 · x_{t1} + σ·√(t2(t1-t2)/t1) · ε
#   Derived from the conditional distribution B_{t2}|B_{t1} of a Brownian bridge.

const DEVICE      = CUDA.functional() ? gpu : cpu
const SEED        = 42
const SAVE_DIR    = "output/sde-out"
const T_EMBED_DIM = 32
const FREQS       = (2f0 .^ Float32.(0:T_EMBED_DIM÷2-1)) |> DEVICE
const σ_BRIDGE    = 0.387f0
const T_EPS       = 1f-3
const N_STEPS     = 200

# --- Data ---

# Source distribution U([2,3]²), disjoint from cat data (≈[-2,2]²)
random_source(n::Int) = rand(Float32, n, 2) .+ 2f0

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

# Residual MLP: time and state branches fused by addition
# Output: x̂_0 = x_t + correction·(t + 0.05)  (residual vanishes as t->0)
struct Net
    time_embed  :: Dense
    state_embed :: Dense
    res         :: Vector{Dense}
    decode      :: Dense
end
Flux.@layer Net

function build_model(data_dim=2, hidden=256, n_res=3)
    Net(
        Dense(T_EMBED_DIM, hidden, gelu),
        Dense(data_dim,    hidden, gelu),
        [Dense(hidden, hidden, gelu) for _ in 1:n_res],
        Dense(hidden, data_dim),
    )
end

function (m::Net)(te, xt)
    h = m.time_embed(te) .+ m.state_embed(xt)
    for layer in m.res
        h = h .+ layer(h)
    end
    return m.decode(h)
end

function t_embed(t::AbstractVector)
    te = reshape(Float32.(t) |> DEVICE, :, 1) .* reshape(FREQS, 1, :)
    return hcat(sin.(te), cos.(te))
end

function model_forward(model, x_t, t)
    te         = t_embed(t)'
    xt         = x_t'
    correction = model(te, xt)'
    t_f        = reshape(Float32.(t) |> DEVICE, :, 1)
    return x_t .+ correction .* (t_f .+ 0.05f0)
end

# --- Bridge ---

# x_t = (1-t)·x_data + t·x_noise + σ·√(t(1-t))·ε
function bridge_forward(x_data::AbstractMatrix, x_noise::AbstractMatrix, t::AbstractVector)
    t_f = reshape(Float32.(t) |> DEVICE, :, 1)
    st  = σ_BRIDGE .* sqrt.(t_f .* (1f0 .- t_f))
    ε   = randn!(similar(x_data))
    return (1f0 .- t_f) .* x_data .+ t_f .* x_noise .+ st .* ε
end

# Exact bridge step: sample x_{t2} | x_{t1}, x̂_0  (t2 < t1, denoising direction)
function bridge_step(x_t::AbstractMatrix, x0_hat::AbstractMatrix, t1::Float32, t2::Float32)
    denom = t1 + 1f-8
    c_x0  = (t1 - t2) / denom
    c_xt  = t2 / denom
    std   = σ_BRIDGE * sqrt(max(t2 * (t1 - t2) / denom, 0f0))
    noise = std > 0f0 ? std .* randn!(similar(x_t)) : zero(x_t)
    return c_x0 .* x0_hat .+ c_xt .* x_t .+ noise
end

# --- Training ---

function warmup!(model, opt_state)
    print("JIT warmup... ")
    t_w     = time()
    x_data  = random_literal_cat(32) |> DEVICE
    x_noise = random_source(32) |> DEVICE
    t       = T_EPS .+ rand(Float32, 32) .* (1f0 - 2f0*T_EPS)
    _, grads = Flux.withgradient(model) do m
        xt      = bridge_forward(x_data, x_noise, t)
        x0_pred = model_forward(m, xt, t)
        scale   = reshape(1f0 ./ (Float32.(t) .+ 0.05f0).^2 |> DEVICE, :, 1)
        mean(scale .* (x0_pred .- x_data).^2)
    end
    Flux.update!(opt_state, model, grads[1])
    CUDA.functional() && CUDA.synchronize()
    @printf("done (%.1fs)\n", time() - t_w)
end

function train(; epochs=5000, batch_size=2048, lr=3e-3, log_interval=500)
    model     = build_model() |> DEVICE
    opt_state = Flux.setup(Adam(lr), model)

    warmup!(model, opt_state)
    println("Training ($epochs steps)...")
    t_start = time()

    for epoch in 1:epochs
        x_data  = random_literal_cat(batch_size) |> DEVICE
        x_noise = random_source(batch_size) |> DEVICE
        t       = T_EPS .+ rand(Float32, batch_size) .* (1f0 - 2f0*T_EPS)

        loss, grads = Flux.withgradient(model) do m
            xt      = bridge_forward(x_data, x_noise, t)
            x0_pred = model_forward(m, xt, t)
            # loss weight 1/(t+0.05)²: upweights steps near t=0 (data end)
            scale   = reshape(1f0 ./ (Float32.(t) .+ 0.05f0).^2 |> DEVICE, :, 1)
            mean(scale .* (x0_pred .- x_data).^2)
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

# --- Sampling ---

function bridge_sample(model, n_samples::Int; n_steps=N_STEPS, verbose=false)
    x  = random_source(n_samples) |> DEVICE
    ts = Float32.(range(1f0, 0f0, length=n_steps+1))

    CUDA.functional() && CUDA.synchronize()
    t0 = time()

    for i in 1:n_steps
        t1, t2 = ts[i], ts[i+1]
        t_vec  = fill(t1, n_samples)
        x0_hat = model_forward(model, x, t_vec)
        x      = bridge_step(x, x0_hat, t1, t2)
    end

    CUDA.functional() && CUDA.synchronize()
    verbose && @printf("  Bridge | steps=%4d  time=%.3fs\n", n_steps, time() - t0)
    return Array(x)
end

function bridge_sample_with_traj(model, n_samples::Int; n_steps=N_STEPS, x_init=nothing)
    x    = isnothing(x_init) ? random_source(n_samples) |> DEVICE : x_init |> DEVICE
    ts   = Float32.(range(1f0, 0f0, length=n_steps+1))
    traj = Vector{Matrix{Float32}}(undef, n_steps + 1)
    traj[1] = Array(x)

    for i in 1:n_steps
        t1, t2 = ts[i], ts[i+1]
        t_vec  = fill(t1, n_samples)
        x0_hat = model_forward(model, x, t_vec)
        x      = bridge_step(x, x0_hat, t1, t2)
        traj[i + 1] = Array(x)
    end

    return Array(x), traj
end

# --- Visualization ---

function visualize_cat(model; n_samples=3000, n_steps=N_STEPS)
    x_gen = bridge_sample(model, n_samples; n_steps=n_steps, verbose=true)
    x_gt  = random_literal_cat(n_samples)

    n_traj  = 20
    x1_traj = random_source(n_traj)
    _, traj    = bridge_sample_with_traj(model, n_traj; n_steps=n_steps, x_init=x1_traj)
    n_traj_pts = length(traj)

    fig = Figure(size=(800, 400))

    ax1 = Axis(fig[1, 1];
               title="Generated vs Reference (Brownian Bridge)",
               xlabel="x1", ylabel="x2", aspect=DataAspect(),
               limits=(-3f0, 3f0, -3f0, 3f0))
    scatter!(ax1, x_gt[:,1],  x_gt[:,2];  markersize=3, color=(:steelblue, 0.25), label="Reference")
    scatter!(ax1, x_gen[:,1], x_gen[:,2]; markersize=3, color=(:tomato,    0.5),  label="Generated")
    axislegend(ax1; position=:lt, labelsize=9)

    ax2 = Axis(fig[1, 2];
               title="Particle Trajectories (noise t=1 -> data t=0)",
               xlabel="x1", ylabel="x2", aspect=DataAspect(),
               limits=(-3f0, 4f0, -3f0, 4f0))
    scatter!(ax2, x_gt[:,1], x_gt[:,2]; markersize=2, color=(:steelblue, 0.15))
    x_end = traj[end]
    for i in 1:n_traj
        xs = [traj[s][i, 1] for s in 1:n_traj_pts]
        ys = [traj[s][i, 2] for s in 1:n_traj_pts]
        lines!(ax2, xs, ys; color=LinRange(0f0, 1f0, n_traj_pts),
               colormap=(:plasma, 0.6), linewidth=0.8)
    end
    scatter!(ax2, x1_traj[:,1], x1_traj[:,2];
             markersize=6, color=:black,  marker=:circle, label="Start (noise)")
    scatter!(ax2, x_end[:,1],   x_end[:,2];
             markersize=6, color=:tomato, marker=:circle, label="End (data)")
    axislegend(ax2; position=:lt, labelsize=9)

    mkpath(SAVE_DIR)
    path = joinpath(SAVE_DIR, "brown_cat.png")
    save(path, fig)
    println("Saved: $path")
end

# --- Main ---

Random.seed!(SEED)
CUDA.functional() && CUDA.seed!(SEED)
println("Using device: " * (CUDA.functional() ? "CUDA GPU" : "CPU"))
println("Starting Brownian Bridge Flow Matching (2D Cat)...")
model = train(epochs=8000, batch_size=2048, lr=3e-3)
visualize_cat(model)
