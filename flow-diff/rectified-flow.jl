using Flux, Zygote, CUDA, cuDNN, OrdinaryDiffEq, Random, Statistics, CairoMakie, Printf

# Rectified Flow (Liu et al. 2022)
# Forward:  x_t = t·x_noise + (1-t)·x_data,  t ~ U[0,1]
# Target:   v* = x_noise - x_data  (straight path -> constant velocity)
# Loss:     E‖v_θ(x_t, t) - v*‖²
# Sampling: ODE dx/dt = v_θ(x,t), integrated from t=1 (noise) to t=0 (data)

const DEVICE      = CUDA.functional() ? gpu : cpu
const SAVE_DIR    = "output/flow-out"
const N_STEPS     = 100
const T_EMBED_DIM = 32
const DATA_LIMS   = ((-3.0f0, 3.0f0), (-3.0f0, 3.0f0))
const FREQS       = (2f0 .^ Float32.(0:T_EMBED_DIM÷2-1)) |> DEVICE

# --- Data ---

function make_moons(n::Int; noise::Float32=0.05f0)
    n_half = n ÷ 2
    θ1 = Float32.(range(0, π, length=n_half))
    θ2 = Float32.(range(0, π, length=n - n_half))
    x1 = hcat(cos.(θ1), sin.(θ1))
    x2 = hcat(1f0 .- cos.(θ2), 1f0 .- sin.(θ2) .- 0.5f0)
    data = vcat(x1, x2)
    data .+= noise .* randn(Float32, size(data))
    return data
end

function make_rings(n::Int; n_rings=3, noise::Float32=0.05f0)
    n_per = n ÷ n_rings
    radii = Float32.(range(0.5, 2.2, length=n_rings))
    pts   = Matrix{Float32}(undef, 0, 2)
    for (k, r) in enumerate(radii)
        count = (k == n_rings) ? (n - size(pts, 1)) : n_per
        θ     = 2f0 * Float32(π) .* rand(Float32, count)
        x     = hcat(r .* cos.(θ), r .* sin.(θ))
        x    .+= noise .* randn(Float32, count, 2)
        pts   = vcat(pts, x)
    end
    return pts[shuffle(1:n), :]
end

function _cat_point(t)
    x = -(721*sin(t))/4+196/3*sin(2*t)-86/3*sin(3*t)-131/2*sin(4*t)+477/14*sin(5*t)+27*sin(6*t)-29/2*sin(7*t)+68/5*sin(8*t)+1/10*sin(9*t)+23/4*sin(10*t)-19/2*sin(12*t)-85/21*sin(13*t)+2/3*sin(14*t)+27/5*sin(15*t)+7/4*sin(16*t)+17/9*sin(17*t)-4*sin(18*t)-1/2*sin(19*t)+1/6*sin(20*t)+6/7*sin(21*t)-1/8*sin(22*t)+1/3*sin(23*t)+3/2*sin(24*t)+13/5*sin(25*t)+sin(26*t)-2*sin(27*t)+3/5*sin(28*t)-1/5*sin(29*t)+1/5*sin(30*t)+(2337*cos(t))/8-43/5*cos(2*t)+322/5*cos(3*t)-117/5*cos(4*t)-26/5*cos(5*t)-23/3*cos(6*t)+143/4*cos(7*t)-11/4*cos(8*t)-31/3*cos(9*t)-13/4*cos(10*t)-9/2*cos(11*t)+41/20*cos(12*t)+8*cos(13*t)+2/3*cos(14*t)+6*cos(15*t)+17/4*cos(16*t)-3/2*cos(17*t)-29/10*cos(18*t)+11/6*cos(19*t)+12/5*cos(20*t)+3/2*cos(21*t)+11/12*cos(22*t)-4/5*cos(23*t)+cos(24*t)+17/8*cos(25*t)-7/2*cos(26*t)-5/6*cos(27*t)-11/10*cos(28*t)+1/2*cos(29*t)-1/5*cos(30*t)
    y = -(637*sin(t))/2-188/5*sin(2*t)-11/7*sin(3*t)-12/5*sin(4*t)+11/3*sin(5*t)-37/4*sin(6*t)+8/3*sin(7*t)+65/6*sin(8*t)-32/5*sin(9*t)-41/4*sin(10*t)-38/3*sin(11*t)-47/8*sin(12*t)+5/4*sin(13*t)-41/7*sin(14*t)-7/3*sin(15*t)-13/7*sin(16*t)+17/4*sin(17*t)-9/4*sin(18*t)+8/9*sin(19*t)+3/5*sin(20*t)-2/5*sin(21*t)+4/3*sin(22*t)+1/3*sin(23*t)+3/5*sin(24*t)-3/5*sin(25*t)+6/5*sin(26*t)-1/5*sin(27*t)+10/9*sin(28*t)+1/3*sin(29*t)-3/4*sin(30*t)-(125*cos(t))/2-521/9*cos(2*t)-359/3*cos(3*t)+47/3*cos(4*t)-33/2*cos(5*t)-5/4*cos(6*t)+31/8*cos(7*t)+9/10*cos(8*t)-119/4*cos(9*t)-17/2*cos(10*t)+22/3*cos(11*t)+15/4*cos(12*t)-5/2*cos(13*t)+19/6*cos(14*t)+7/4*cos(15*t)+31/4*cos(16*t)-cos(17*t)+11/10*cos(18*t)-2/3*cos(19*t)+13/3*cos(20*t)-5/4*cos(21*t)+2/3*cos(22*t)+1/4*cos(23*t)+5/6*cos(24*t)+3/4*cos(26*t)-1/2*cos(27*t)-1/10*cos(28*t)-1/3*cos(29*t)-1/19*cos(30*t)
    return Float32[x, y]
end

function make_cat(n::Int; noise::Float32=0.05f0)
    pts = reduce(hcat, [_cat_point(rand() * 2f0 * Float32(π)) for _ in 1:n])
    pts ./= 200f0
    pts .+= randn(Float32, 2, n) .* noise
    return Matrix(pts')
end

function make_checkerboard(n::Int; noise::Float32=0.02f0)
    pts = Matrix{Float32}(undef, 0, 2)
    while size(pts, 1) < n
        batch = 4n
        x    = rand(Float32, batch, 2) .* 4f0 .- 2f0
        ix   = floor.(Int, x[:, 1] .+ 2f0)
        iy   = floor.(Int, x[:, 2] .+ 2f0)
        mask = (ix .+ iy) .% 2 .== 0
        pts  = vcat(pts, x[mask, :])
    end
    idx = shuffle(1:size(pts, 1))[1:n]
    pts = pts[idx, :]
    pts .+= noise .* randn(Float32, n, 2)
    return pts
end

# --- Model ---

# Sinusoidal time embedding: scalar t -> T_EMBED_DIM-dimensional vector
function t_embed(t::AbstractVector)
    te = reshape(t, :, 1) .* reshape(FREQS, 1, :)
    return hcat(sin.(te), cos.(te))
end

# Residual MLP: time and state branches fused by addition
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

function model_forward(model, x_t, t)
    te = t_embed(t)'
    xt = x_t'
    return model(te, xt)'
end

# --- Core utilities ---

# Source distribution U([2,3]²), disjoint from cat data (≈[-2,2]²)
random_source(n::Int) = (CUDA.functional() ? CUDA.rand(Float32, n, 2) : rand(Float32, n, 2)) .+ 2f0

# Linear interpolation: x_t = t·x_noise + (1-t)·x_data  (t=1: noise, t=0: data)
function interp(x_data, x_noise, t::AbstractVector)
    t_ = reshape(t, :, 1)
    return t_ .* x_noise .+ (1f0 .- t_) .* x_data
end

# --- Samplers ---

# Euler: 1st-order fixed-step, NFE = n_steps
function rf_sample(model, n_samples::Int; n_steps=N_STEPS)
    x     = random_source(n_samples) |> DEVICE
    dt    = Float32(1.0 / n_steps)
    t_vec = zeros(Float32, n_samples) |> DEVICE
    for i in 0:n_steps-1
        fill!(t_vec, 1f0 - Float32(i) / Float32(n_steps))
        v = model_forward(model, x, t_vec)
        x = x .- dt .* v
    end
    return x |> cpu
end

# Euler with timing; returns (samples, nfe, elapsed)
function euler_sample(model, n_samples::Int; n_steps=N_STEPS, verbose=false)
    x     = random_source(n_samples) |> DEVICE
    dt    = Float32(1.0 / n_steps)
    t_vec = zeros(Float32, n_samples) |> DEVICE

    CUDA.functional() && CUDA.synchronize()
    t_start = time()
    for i in 0:n_steps-1
        fill!(t_vec, 1f0 - Float32(i) / Float32(n_steps))
        v = model_forward(model, x, t_vec)
        x = x .- dt .* v
    end
    CUDA.functional() && CUDA.synchronize()
    elapsed = time() - t_start

    verbose && @printf("Euler       steps=%4d  NFE=%4d  time=%.3fs\n", n_steps, n_steps, elapsed)
    return x |> cpu, n_steps, elapsed
end

# Heun: 2nd-order predictor-corrector, NFE = 2·n_steps; returns (samples, nfe, elapsed)
function heun_sample(model, n_samples::Int; n_steps=N_STEPS, verbose=false)
    x      = random_source(n_samples) |> DEVICE
    dt     = Float32(1.0 / n_steps)
    t_vec0 = zeros(Float32, n_samples) |> DEVICE
    t_vec1 = zeros(Float32, n_samples) |> DEVICE
    nfe    = 0

    CUDA.functional() && CUDA.synchronize()
    t_start = time()
    for i in 0:n_steps-1
        fill!(t_vec0, 1f0 - Float32(i)     / Float32(n_steps))
        fill!(t_vec1, 1f0 - Float32(i + 1) / Float32(n_steps))
        k1 = model_forward(model, x,             t_vec0);  nfe += 1
        k2 = model_forward(model, x .- dt .* k1, t_vec1);  nfe += 1
        x  = x .- (dt / 2f0) .* (k1 .+ k2)
    end
    CUDA.functional() && CUDA.synchronize()
    elapsed = time() - t_start

    verbose && @printf("Heun        steps=%4d  NFE=%4d  time=%.3fs\n", n_steps, nfe, elapsed)
    return x |> cpu, nfe, elapsed
end

# Tsit5 fixed-step (~4th-order): NFE ≈ 6·n_steps; returns (samples, nfe, elapsed)
function rk4_sample(model, n_samples::Int; n_steps=20, verbose=false)
    x0    = random_source(n_samples) |> DEVICE
    nfe   = Ref(0)
    dt    = 1f0 / n_steps
    t_vec = zeros(Float32, n_samples) |> DEVICE

    function f!(du, u, p, t)
        fill!(t_vec, Float32(t))
        v    = model_forward(model, u, t_vec)
        du  .= v
        p[] += 1
    end

    prob = ODEProblem(f!, x0, (1f0, 0f0), nfe)

    CUDA.functional() && CUDA.synchronize()
    t_start = time()
    sol = solve(prob, Tsit5(); adaptive=false, dt=-dt, save_everystep=false)
    CUDA.functional() && CUDA.synchronize()
    elapsed = time() - t_start

    verbose && @printf("Tsit5-fixed steps=%4d  NFE=%4d  time=%.3fs\n", n_steps, nfe[], elapsed)
    return Array(sol.u[end]), nfe[], elapsed
end

# Tsit5 adaptive with initial step hint; returns (samples, nfe, elapsed)
function adaptive_ode_sample_dt(model, n_samples::Int;
                                 n_steps_hint=N_STEPS,
                                 abstol=1f-3, reltol=1f-3, verbose=false)
    x0    = random_source(n_samples) |> DEVICE
    nfe   = Ref(0)
    t_vec = zeros(Float32, n_samples) |> DEVICE
    dt0   = -1f0 / Float32(n_steps_hint)

    function f!(du, u, p, t)
        fill!(t_vec, Float32(t))
        v   = model_forward(model, u, t_vec)
        du .= v
        p[] += 1
    end

    prob = ODEProblem(f!, x0, (1f0, 0f0), nfe)

    CUDA.functional() && CUDA.synchronize()
    t_start = time()
    sol = solve(prob, Tsit5(); dt=dt0, abstol=abstol, reltol=reltol, save_everystep=false)
    CUDA.functional() && CUDA.synchronize()
    elapsed = time() - t_start

    verbose && @printf("Tsit5-adap  hint=%4d  NFE=%4d  tol=%.0e  time=%.3fs\n",
                       n_steps_hint, nfe[], abstol, elapsed)
    return Array(sol.u[end]), nfe[], elapsed
end

# Euler with trajectory recording; returns (x_final, traj)
function rf_sample_with_traj(model, n_particles::Int; n_steps=N_STEPS)
    x     = random_source(n_particles) |> DEVICE
    dt    = Float32(1.0 / n_steps)
    t_vec = zeros(Float32, n_particles) |> DEVICE
    traj  = Vector{Matrix{Float32}}(undef, n_steps + 1)
    traj[1] = x |> cpu
    for i in 0:n_steps-1
        fill!(t_vec, 1f0 - Float32(i) / Float32(n_steps))
        v           = model_forward(model, x, t_vec)
        x           = x .- dt .* v
        traj[i + 2] = x |> cpu
    end
    return x |> cpu, traj
end

# --- Training ---

function warmup!(model, opt_state; data_fn=make_cat)
    print("JIT warmup... ")
    t_w     = time()
    x_data  = data_fn(32) |> DEVICE
    x_noise = random_source(32)
    t       = CUDA.functional() ? CUDA.rand(Float32, 32) : rand(Float32, 32)
    _, grads = Flux.withgradient(model) do m
        x_t    = interp(x_data, x_noise, t)
        v_pred = model_forward(m, x_t, t)
        v_true = x_noise .- x_data
        mean((v_pred .- v_true).^2)
    end
    Flux.update!(opt_state, model, grads[1])
    CUDA.functional() && CUDA.synchronize()
    @printf("done (%.1fs)\n", time() - t_w)
end

function train_flow(; data_fn=make_cat, epochs=8000, batch_size=4096, lr=3e-3, log_interval=500)
    model     = build_model() |> DEVICE
    opt_state = Flux.setup(Adam(lr), model)
    losses    = Float32[]

    warmup!(model, opt_state; data_fn=data_fn)

    t_total_start  = time()
    t_window_start = time()

    for epoch in 1:epochs
        x_data  = data_fn(batch_size) |> DEVICE
        x_noise = random_source(batch_size)
        t       = CUDA.functional() ? CUDA.rand(Float32, batch_size) : rand(Float32, batch_size)

        loss, grads = Flux.withgradient(model) do m
            x_t    = interp(x_data, x_noise, t)
            v_pred = model_forward(m, x_t, t)
            v_true = x_noise .- x_data
            mean((v_pred .- v_true).^2)
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
    end

    total = time() - t_total_start
    @printf("\nTraining done: %.2fs total, %.1f iter/s average\n", total, epochs / total)
    return model, losses
end

# --- Metrics ---

# Chamfer Distance: mean bidirectional nearest-neighbor distance, lower is better
function chamfer_distance(a::Matrix{Float32}, b::Matrix{Float32})::Float32
    na, nb = size(a, 1), size(b, 1)
    diff  = reshape(a, na, 1, 2) .- reshape(b, 1, nb, 2)
    dists = sqrt.(sum(diff .^ 2; dims=3))[:, :, 1]
    d_ab  = mean(minimum(dists; dims=2))
    d_ba  = mean(minimum(dists; dims=1))
    return Float32(d_ab + d_ba)
end

# MMD²: RBF kernel (σ=0.5), equals 0 for identical distributions, lower is better
function mmd_rbf(a::Matrix{Float32}, b::Matrix{Float32}; sigma::Float32=0.5f0)::Float32
    function rbf_mean(x::Matrix{Float32}, y::Matrix{Float32})
        nx, ny = size(x, 1), size(y, 1)
        diff = reshape(x, nx, 1, 2) .- reshape(y, 1, ny, 2)
        sq   = sum(diff .^ 2; dims=3)[:, :, 1]
        return mean(exp.(-sq ./ (2f0 * sigma^2)))
    end
    return rbf_mean(a, a) - 2f0 * rbf_mean(a, b) + rbf_mean(b, b)
end

# --- Visualization ---

function visualize_cat(model, x_gt; n_gen=3000, n_traj=20, n_steps=N_STEPS)
    x_gen = rf_sample(model, n_gen; n_steps=n_steps)
    _, traj = rf_sample_with_traj(model, n_traj; n_steps=n_steps)
    n_snap = length(traj)

    fig = Figure(size=(800, 400))

    ax1 = Axis(fig[1, 1],
               title="Generated vs Reference (Rectified Flow)",
               xlabel="x1", ylabel="x2",
               limits=(-3f0, 3f0, -3f0, 3f0))
    scatter!(ax1, x_gt[:,1],  x_gt[:,2];  markersize=3, color=(:steelblue, 0.25), label="Reference")
    scatter!(ax1, x_gen[:,1], x_gen[:,2]; markersize=3, color=(:tomato,    0.5),  label="Generated")
    axislegend(ax1; position=:lt, labelsize=9)

    ax2 = Axis(fig[1, 2],
               title="Particle Trajectories (noise t=1 -> data t=0)",
               xlabel="x1", ylabel="x2", aspect=DataAspect(),
               limits=(-3f0, 4f0, -3f0, 4f0))
    scatter!(ax2, x_gt[:,1], x_gt[:,2]; markersize=3, color=(:steelblue, 0.15))
    for p in 1:n_traj
        xs = [traj[s][p, 1] for s in 1:n_snap]
        ys = [traj[s][p, 2] for s in 1:n_snap]
        lines!(ax2, xs, ys; color=LinRange(0f0, 1f0, n_snap), colormap=(:plasma, 0.6), linewidth=0.9)
    end
    x0    = traj[1]
    x_end = traj[end]
    scatter!(ax2, x0[:,1],    x0[:,2];    markersize=6, color=:black,  marker=:circle, label="Start (noise)")
    scatter!(ax2, x_end[:,1], x_end[:,2]; markersize=6, color=:tomato, marker=:circle, label="End (data)")
    axislegend(ax2; position=:lt, labelsize=9)

    path = joinpath(SAVE_DIR, "ode_cat.png")
    save(path, fig)
    println("Saved: $path")
end

# 4-sampler comparison (Euler / Heun / Tsit5-fixed / Tsit5-adaptive) across step_list steps
function visualize_sampler_comparison(model, x_gt::Matrix{Float32};
                                       n_samples=2000,
                                       step_list=[10, 20, 50, 100],
                                       tol_list=[5f-2, 1f-2, 1f-3, 5f-4])
    n_col = length(step_list)

    print("JIT warmup... ")
    for fn in (euler_sample, heun_sample, rk4_sample)
        fn(model, 64; n_steps=5)
    end
    adaptive_ode_sample_dt(model, 64; n_steps_hint=5, abstol=1f-1, reltol=1f-1)
    CUDA.functional() && CUDA.synchronize()
    println("done")

    res_euler = NamedTuple[]
    res_heun  = NamedTuple[]
    res_rk4   = NamedTuple[]
    res_tsit5 = NamedTuple[]

    println("\n-- Sampler Comparison on Cat Distribution --")
    @printf("%-22s  %6s  %6s  %10s  %10s\n", "method", "steps", "NFE", "CD↓", "MMD²↓")
    println("-"^60)

    for (n_steps, tol) in zip(step_list, tol_list)
        xe, nfe_e, _ = euler_sample(model, n_samples; n_steps=n_steps)
        xh, nfe_h, _ = heun_sample(model, n_samples; n_steps=n_steps)
        xr, nfe_r, _ = rk4_sample(model, n_samples; n_steps=n_steps)
        xt, nfe_t, _ = adaptive_ode_sample_dt(model, n_samples;
                                               n_steps_hint=n_steps, abstol=tol, reltol=tol)

        cd_e, mmd_e = chamfer_distance(xe, x_gt), mmd_rbf(xe, x_gt)
        cd_h, mmd_h = chamfer_distance(xh, x_gt), mmd_rbf(xh, x_gt)
        cd_r, mmd_r = chamfer_distance(xr, x_gt), mmd_rbf(xr, x_gt)
        cd_t, mmd_t = chamfer_distance(xt, x_gt), mmd_rbf(xt, x_gt)

        @printf("Euler          %6d  %6d  %10.4f  %10.6f\n",              n_steps, nfe_e, cd_e, mmd_e)
        @printf("Heun           %6d  %6d  %10.4f  %10.6f\n",              n_steps, nfe_h, cd_h, mmd_h)
        @printf("Tsit5-fixed    %6d  %6d  %10.4f  %10.6f\n",              n_steps, nfe_r, cd_r, mmd_r)
        @printf("Tsit5-adaptive %6d  %6d  %10.4f  %10.6f  (tol=%.0e)\n", n_steps, nfe_t, cd_t, mmd_t, tol)

        push!(res_euler, (; n_steps, nfe=nfe_e, cd=cd_e, mmd=mmd_e, samples=xe))
        push!(res_heun,  (; n_steps, nfe=nfe_h, cd=cd_h, mmd=mmd_h, samples=xh))
        push!(res_rk4,   (; n_steps, nfe=nfe_r, cd=cd_r, mmd=mmd_r, samples=xr))
        push!(res_tsit5, (; n_steps, nfe=nfe_t, cd=cd_t, mmd=mmd_t, samples=xt, tol))
    end

    function make_step_fig(title_str, results, dot_color; extra_fn=nothing)
        fig = Figure(size=(n_col * 380, 480))
        Label(fig[0, 1:n_col], title_str; fontsize=12, font=:bold)
        for (col, r) in enumerate(results)
            extra = isnothing(extra_fn) ? "" : extra_fn(r)
            ax = Axis(fig[1, col];
                      title  = "$(r.n_steps) steps  NFE=$(r.nfe)$(extra)\nCD=$(round(r.cd, digits=4))  MMD²=$(round(r.mmd, digits=5))",
                      xlabel = "x₁", ylabel = "x₂",
                      limits = DATA_LIMS)
            scatter!(ax, x_gt[:,1],      x_gt[:,2];      markersize=3, color=(:steelblue, 0.25), label="GT")
            scatter!(ax, r.samples[:,1], r.samples[:,2]; markersize=3, color=(dot_color,  0.55), label="Gen")
            axislegend(ax; position=:lt, labelsize=8)
        end
        return fig
    end

    fig1 = make_step_fig("Euler  (fixed step · 1st-order · NFE = N)",
                         res_euler, :tomato)
    fig2 = make_step_fig("Heun  (fixed step · 2nd-order · NFE = 2N)",
                         res_heun, :mediumseagreen)
    fig3 = make_step_fig("Tsit5 fixed-step  (4th-order · NFE ≈ 6N)",
                         res_rk4, :darkorange)
    fig4 = make_step_fig("Tsit5 adaptive  (initial dt = 1/N, tol-controlled)",
                         res_tsit5, :mediumpurple;
                         extra_fn=r -> @sprintf("  tol=%.0e", r.tol))

    for (fname, fig) in (("cat_euler.png",       fig1),
                          ("cat_heun.png",        fig2),
                          ("cat_tsit5_fixed.png", fig3),
                          ("cat_tsit5_adap.png",  fig4))
        path = joinpath(SAVE_DIR, fname)
        save(path, fig)
        println("Saved: $path")
    end
end

function plot_loss(losses)
    fig = Figure()
    ax  = Axis(fig[1, 1], xlabel="Epoch", ylabel="MSE Loss",
               title="Rectified Flow Training Loss", yscale=log10)
    lines!(ax, losses, color=:black)
    path = joinpath(SAVE_DIR, "jl_rf_loss.png")
    save(path, fig)
    println("Saved: $path")
end

# --- Main ---

println("Using device: " * (CUDA.functional() ? "CUDA GPU" : "CPU"))
mkpath(SAVE_DIR)

const DATA_FN = make_cat   # alternatives: make_rings, make_moons, make_checkerboard

println("Training Rectified Flow ($(nameof(DATA_FN)))...")
model, losses = train_flow(data_fn=DATA_FN, epochs=8000, batch_size=2048, lr=3e-3)
println("Training done!")
plot_loss(losses)

x_gt = DATA_FN(2000)
visualize_cat(model, x_gt; n_gen=3000, n_traj=20, n_steps=100)

visualize_sampler_comparison(model, x_gt;
    n_samples = 2000,
    step_list = [10, 20, 50, 100],
    tol_list  = [5f-2, 1f-2, 1f-3, 5f-4])
