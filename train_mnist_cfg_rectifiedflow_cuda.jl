# =========================
# MNIST Conditional Rectified Flow + CFG (CUDA)
# File: examples/train_mnist_cfg_rectifiedflow_cuda.jl
# Run:
#   $env:JULIA_CUDA_USE_BINARYBUILDER="true"
#   julia --project=. examples/train_mnist_cfg_rectifiedflow_cuda.jl
# =========================

import Pkg
using Random, Printf, Dates, Statistics

# --- activate project root (one level up from examples/)
PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)

# --- ensure deps
function ensure_pkg(name::String)
    if Base.find_package(name) === nothing
        @info "Installing missing package: $name"
        Pkg.add(name)
    end
end

ensure_pkg("CUDA")
ensure_pkg("cuDNN")
ensure_pkg("MLDatasets")
ensure_pkg("Flux")
ensure_pkg("ProgressMeter")
ensure_pkg("JSON")
ensure_pkg("BSON")
ensure_pkg("Plots")
ensure_pkg("GR")
Pkg.instantiate()

using Statistics
using CUDA
using cuDNN
using Flux
using MLDatasets
using ProgressMeter
using JSON, BSON
using Plots

ENV["GKSwstype"] = "100"  # headless save png

CUDA.allowscalar(false)
@info "CUDA functional? => $(CUDA.functional())"
CUDA.versioninfo()
CUDA.functional() || error("CUDA not functional. Fix CUDA.jl/driver first.")

const to_device = gpu
const USE_GPU = true

# =========================
# Settings
# =========================
seed           = 2714
batch_size     = 128
steps          = 20000               # quick run; use 50k~200k for better quality
lr             = 2e-4
log_interval   = 200
save_interval  = 2000

prob_uncond    = 0.20f0              # CFG training: 20% drop label
guidance_scale = 2.0f0               # CFG sampling scale
n_sample_each  = 8                   # samples per class (10 classes -> 80 total)
n_ode_steps    = 50                  # ODE steps for sampling

num_digits     = 10
num_classes    = num_digits + 1      # 0=uncond, 1..10 = digit(0..9)

hidden         = 64

output_directory = joinpath("outputs", "MNIST_CFG_RF_CUDA_" * Dates.format(now(), "yyyymmdd_HHMM"))
mkpath(output_directory)
@info "Output dir: $output_directory"

Random.seed!(seed)

# =========================
# Data
# =========================
function load_mnist_train()
    x, y = MNIST.traindata(Float32)          # x: 28×28×N
    x = reshape(x, 28, 28, 1, size(x, 3))    # WHCN
    x = x ./ 255f0
    x = 2f0 .* x .- 1f0                      # [-1,1]
    y = Int.(vec(y))                         # 0..9
    return x, y
end

x_all, y_all = load_mnist_train()
N = size(x_all, 4)

# split train/val (90/10)
perm = randperm(MersenneTwister(seed), N)
n_train = Int(floor(0.9N))
tr_idx = perm[1:n_train]
va_idx = perm[n_train+1:end]

x_train = x_all[:, :, :, tr_idx]
y_train = y_all[tr_idx]

x_val   = x_all[:, :, :, va_idx]
y_val   = y_all[va_idx]

println("train data:      ", size(x_train), " labels: ", size(y_train))
println("validation data: ", size(x_val),   " labels: ", size(y_val))

# =========================
# Rectified Flow forward/target
# x_t = t*x_noise + (1-t)*x_data ; t ~ U[0,1]
# v*  = x_noise - x_data
# =========================
random_source_like(x0_cpu) = randn(Float32, size(x0_cpu))

function interp(x_data, x_noise, t::AbstractVector)
    t_ = reshape(Float32.(t), 1, 1, 1, :)
    return t_ .* x_noise .+ (1f0 .- t_) .* x_data
end

# =========================
# Conditioning encoding (one-hot planes)
# channels: [x_t(1), t(1), onehot_class(num_classes)]
# total in_ch = 1 + 1 + num_classes
# label coding:
#   cls=0 => unconditional token
#   cls=1..10 => digit = cls-1
# =========================
function class_planes(cls::Vector{Int}, H::Int, W::Int, C::Int)
    # Zygote-safe, non-mutating one-hot planes
    # cls values: 0..(C-1)
    B = length(cls)
    c_idx = reshape(collect(0:C-1), 1, 1, C, 1)   # 1×1×C×1
    cls_r = reshape(cls,            1, 1, 1, B)   # 1×1×1×B
    return Float32.(c_idx .== cls_r) .* ones(Float32, H, W, 1, 1)  # H×W×C×B
end

function add_t_and_class(x_t, t::AbstractVector, cls::Vector{Int})
    B = size(x_t, 4)
    t_ = reshape(Float32.(t), 1, 1, 1, B)
    t_img = ones(Float32, 28, 28, 1, B) .* t_
    c_img = class_planes(cls, 28, 28, num_classes)
    return cat(x_t, t_img, c_img; dims=3)   # 28×28×(1+1+num_classes)×B
end

# =========================
# Model: small CNN
# =========================
function build_model()
    in_ch = 1 + 1 + num_classes
    Chain(
        Conv((3,3), in_ch => hidden, pad=1, gelu),
        Conv((3,3), hidden => hidden, pad=1, gelu),
        Conv((3,3), hidden => hidden, pad=1, gelu),
        Conv((3,3), hidden => 1, pad=1),
    )
end

model = build_model() |> to_device
opt_state = Flux.setup(Adam(lr), model)

# =========================
# Loss
# =========================
function rf_loss(model, x0_cpu, y_digit::Vector{Int})
    B = length(y_digit)

    # CFG training: randomly drop label -> unconditional
    mask = rand(Float32, B) .< prob_uncond
    cls  = [mask[i] ? 0 : (y_digit[i] + 1) for i in 1:B]  # 0=uncond, 1..10=digits

    t  = rand(Float32, B)                     # [0,1]
    x1 = random_source_like(x0_cpu)           # noise (CPU)
    xt = interp(x0_cpu, x1, t)                # CPU

    x0 = x0_cpu |> to_device
    x1 = x1 |> to_device
    xt = xt |> to_device

    v_target = x1 .- x0                       # GPU

    inp = add_t_and_class(xt |> cpu, t, cls) |> to_device
    v_pred = model(inp)

    return mean((v_pred .- v_target).^2)
end

# =========================
# Validation (no CFG drop)
# =========================
function rf_val_loss(model, x0_cpu, y_digit::Vector{Int})
    B = length(y_digit)
    cls = [d + 1 for d in y_digit]           # always conditional
    t  = rand(Float32, B)
    x1 = random_source_like(x0_cpu)
    xt = interp(x0_cpu, x1, t)

    x0 = x0_cpu |> to_device
    x1 = x1 |> to_device
    xt = xt |> to_device

    v_target = x1 .- x0
    inp = add_t_and_class(xt |> cpu, t, cls) |> to_device
    v_pred = model(inp)
    return mean((v_pred .- v_target).^2)
end

# =========================
# Sampling with CFG
# v = v_u + s*(v_c - v_u)
# ODE Euler from t=1 -> 0
# =========================
function sample_cfg(model; digit::Int, n::Int=8, n_steps::Int=50, scale::Float32=2f0)
    x = randn(Float32, 28, 28, 1, n) |> to_device
    dt = Float32(1f0 / n_steps)

    t_vec = zeros(Float32, n)
    cls_u = fill(0, n)              # unconditional
    cls_c = fill(digit + 1, n)      # conditional class

    for i in 0:n_steps-1
        fill!(t_vec, 1f0 - Float32(i)/Float32(n_steps))

        inp_u = add_t_and_class(x |> cpu, t_vec, cls_u) |> to_device
        inp_c = add_t_and_class(x |> cpu, t_vec, cls_c) |> to_device

        v_u = model(inp_u)
        v_c = model(inp_c)
        v   = v_u .+ scale .* (v_c .- v_u)

        x = x .- dt .* v
    end

    x = clamp.(x, -1f0, 1f0) |> cpu
    return x
end

function to_gray_images(x)
    x2 = dropdims(x; dims=3)        # 28×28×N
    n  = size(x2, 3)
    imgs = Vector{Matrix{Float32}}(undef, n)
    for i in 1:n
        img = (x2[:, :, i] .+ 1f0) ./ 2f0
        imgs[i] = clamp.(img, 0f0, 1f0)
    end
    return imgs
end

function save_samples_grid(model, step)
    imgs = Matrix{Float32}[]
    titles = String[]
    for d in 0:9
        x = sample_cfg(model; digit=d, n=n_sample_each, n_steps=n_ode_steps, scale=guidance_scale)
        push!(titles, "digit=$d")
        append!(imgs, to_gray_images(x))
    end

    # plot grid: 10 rows, n_sample_each cols
    plots = []
    idx = 1
    for r in 1:10
        row = []
        for c in 1:n_sample_each
            push!(row, heatmap(imgs[idx], aspect_ratio=1, axis=false))
            idx += 1
        end
        push!(plots, plot(row...; layout=(1, n_sample_each), margin=1Plots.mm, title=titles[r]))
    end
    canvas = plot(plots...; layout=(10,1), size=(1200, 2000))
    path = joinpath(output_directory, "samples_step_$(lpad(step,6,'0')).png")
    savefig(canvas, path)
    println("Saved samples: $path")
end

# =========================
# Training loop
# =========================
function get_batch(x, y, bs::Int)
    n = size(x, 4)
    idx = rand(1:n, bs)
    return x[:, :, :, idx], y[idx]
end

println("Calculating initial validation loss...")
xv, yv = get_batch(x_val, y_val, min(batch_size, 256))
val0 = rf_val_loss(model, xv, yv)
@printf("val loss (init): %.6f\n\n", val0)

loss_hist = Float32[]
val_hist  = Float32[]

println("Starting training on GPU (Rectified Flow + CFG)...")
t0 = time_ns()

p = Progress(steps; dt=0.5, desc="steps")
for s in 1:steps
    xb, yb = get_batch(x_train, y_train, batch_size)

    l, grads = Flux.withgradient(model) do m
        rf_loss(m, xb, yb)
    end
    Flux.update!(opt_state, model, grads[1])
    push!(loss_hist, Float32(l))

    if s % log_interval == 0
        local xv, yv = get_batch(x_val, y_val, min(batch_size, 256))
        vl = rf_val_loss(model, xv, yv)
        push!(val_hist, Float32(vl))
        @printf("step %6d/%d  train=%.6f  val=%.6f\n", s, steps, l, vl)
    end

    if s % save_interval == 0
        save_samples_grid(model, s)
        BSON.bson(joinpath(output_directory, "model_step_$(lpad(s,6,'0')).bson"),
                  Dict(:model => cpu(model)))
    end

    next!(p)
end

elapsed = (time_ns() - t0) / 1e9
@printf("\nTraining done. Time: %.2fs\n", elapsed)

open(joinpath(output_directory, "history.json"), "w") do f
    JSON.print(f, Dict(
        "train_loss" => loss_hist,
        "val_loss" => val_hist,
        "steps" => steps,
        "batch_size" => batch_size,
        "prob_uncond" => prob_uncond,
        "guidance_scale" => guidance_scale,
        "n_ode_steps" => n_ode_steps
    ))
end

save_samples_grid(model, steps)
BSON.bson(joinpath(output_directory, "model_final.bson"), Dict(:model => cpu(model)))

println("\nFinished MNIST Conditional Rectified Flow + CFG (CUDA).")