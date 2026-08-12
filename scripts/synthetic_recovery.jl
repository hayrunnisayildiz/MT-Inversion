#!/usr/bin/env julia
# Synthetic MT recovery test:
#   known ρ(z) → forward → noise → invert from uniform prior
#
#   julia --project=. scripts/synthetic_recovery.jl

include(joinpath(@__DIR__, "..", "src", "SciMLInversion.jl"))
using .SciMLInversion
using .SciMLInversion: DataIngestion, PriorModels, ForwardModel
using Random
using Zygote
using Statistics: mean

"""
Build a known layered true model on a small 3D grid (only the center column matters).
"""
function make_true_model(; nx=3, ny=3, nz=10, default_rho=100.0)
    prior = PriorModels.create_3d_prior_model(;
        nx=nx, ny=ny, nz=nz,
        dx_base=1000.0, dy_base=1000.0,
        dz_base=150.0, z_factor=1.25,
        default_rho=default_rho,
    )
    # Layered truth: host → conductor → resistive basement
    # indices 1..nz (shallow → deep)
    true_col = fill(default_rho, nz)
    true_col[3:5] .= 10.0      # conductive layer
    true_col[7:9] .= 500.0     # resistive basement

    ix = (nx + 1) ÷ 2
    iy = (ny + 1) ÷ 2
    ρ = copy(prior.resistivity)
    for k in 1:nz
        ρ[ix, iy, k] = true_col[k]
    end

    true_model = PriorModels.PriorModel3D(
        prior.nx, prior.ny, prior.nz,
        prior.dx, prior.dy, prior.dz,
        ρ, prior.rho_min, prior.rho_max,
    )
    return true_model, true_col, (ix, iy)
end

"""
Add relative Gaussian noise to each impedance tensor entry.
"""
function add_impedance_noise(Z_clean, noise_frac; rng=Random.default_rng())
    Z_noisy = similar(Z_clean)
    for i in eachindex(Z_clean)
        Z = copy(Z_clean[i])
        for j in eachindex(Z)
            σ = noise_frac * abs(Z[j])
            if σ > 0
                Z[j] += Complex(σ * randn(rng), σ * randn(rng))
            end
        end
        Z_noisy[i] = Z
    end
    return Z_noisy
end

"""
Gradient descent with explicit loss history (for the recovery report).
"""
function invert_with_history(
    station_data,
    prior_model;
    iterations=40,
    lr=1.0,
    lambda=1e-4,
)
    ps = SciMLInversion.build_inversion_model(prior_model)
    Z_obs = station_data.Z
    frequencies = station_data.frequencies

    history = NamedTuple{(:total, :misfit, :reg), Tuple{Float64, Float64, Float64}}[]

    for iter in 1:iterations
        terms = SciMLInversion.loss_terms(ps, Z_obs, frequencies, prior_model; lambda=lambda)
        push!(history, (total=Float64(terms.total), misfit=Float64(terms.misfit), reg=Float64(terms.reg)))

        _, back = Zygote.pullback(
            p -> SciMLInversion.compute_loss(p, Z_obs, frequencies, prior_model; lambda=lambda),
            ps,
        )
        ∇ps = back(one(terms.total))[1]
        ps = ps .- lr .* ∇ps

        if iter == 1 || iter == iterations || iter % 5 == 0
            println(
                "  iter ", lpad(string(iter), 2), "/", iterations,
                "  L = ", round(terms.total; sigdigits=5),
                "  (misfit=", round(terms.misfit; sigdigits=5),
                ", reg=", round(terms.reg; sigdigits=5), ")",
            )
        end
    end

    final = SciMLInversion.loss_terms(ps, Z_obs, frequencies, prior_model; lambda=lambda)
    push!(history, (total=Float64(final.total), misfit=Float64(final.misfit), reg=Float64(final.reg)))
    return ps, history
end

function main()
    Random.seed!(7)

    println("=== Synthetic MT recovery ===")
    println("known ρ(z) → forward → noise → recover from uniform prior")
    println()

    # 1) Known truth
    true_model, true_col, (ix, iy) = make_true_model()
    frequencies = exp10.(range(-2, 2; length=16))  # 0.01 … 100 Hz

    println("── Truth ρ(z) at center column ──")
    z_top = 0.0
    for k in eachindex(true_col)
        z_bot = z_top + true_model.dz[k]
        println(
            "  z ∈ [", round(z_top; digits=0), ", ", round(z_bot; digits=0), "] m",
            "  ρ = ", true_col[k], " Ω·m",
        )
        z_top = z_bot
    end
    println()

    # 2) Forward
    Z_clean = ForwardModel.compute_station_responses(true_model, frequencies)

    # 3) Noise
    noise_frac = 0.03
    Z_obs = add_impedance_noise(Z_clean, noise_frac)
    station_data = DataIngestion.MTData(
        "SYNTHETIC",
        0.0, 0.0, 0.0,
        collect(frequencies),
        Z_obs,
    )
    println("Frequencies : ", length(frequencies))
    println("Noise       : ", 100 * noise_frac, "% relative Gaussian on Z")
    println()

    # 4) Invert from wrong (uniform) prior
    prior = PriorModels.create_3d_prior_model(;
        nx=true_model.nx, ny=true_model.ny, nz=true_model.nz,
        dx_base=true_model.dx[1], dy_base=true_model.dy[1],
        dz_base=true_model.dz[1], z_factor=true_model.dz[2] / true_model.dz[1],
        default_rho=100.0,
    )
    # Match dz exactly (create_3d_prior_model rebuilds geometric dz)
    prior = PriorModels.PriorModel3D(
        true_model.nx, true_model.ny, true_model.nz,
        copy(true_model.dx), copy(true_model.dy), copy(true_model.dz),
        fill(100.0, true_model.nx, true_model.ny, true_model.nz),
        true_model.rho_min, true_model.rho_max,
    )

    println("── Inversion (uniform 100 Ω·m start) ──")
    iterations = 40
    lr = 1.0
    lambda = 1e-4
    ps, history = invert_with_history(
        station_data, prior;
        iterations=iterations, lr=lr, lambda=lambda,
    )

    L0 = history[1].total
    Lf = history[end].total
    drop = L0 - Lf
    drop_pct = 100 * drop / L0

    println()
    println("── Loss check ──")
    println("  L_initial = ", L0)
    println("  L_final   = ", Lf)
    println("  ΔL        = ", drop, "  (", round(drop_pct; digits=2), "% drop)")

    if !(Lf < L0)
        error("FAIL: loss did not decrease (L0=$L0, Lf=$Lf)")
    end
    println("  ✓ Loss decreased")

    # Recovered center column
    ρ_rec = exp.(ps.log_rho)
    rec_col = [ρ_rec[ix, iy, k] for k in 1:true_model.nz]
    rel_err = mean(abs.(rec_col .- true_col) ./ true_col)

    println()
    println("── ρ(z) recovery (center column) ──")
    println("  k    true      recovered")
    for k in eachindex(true_col)
        println(
            "  ", lpad(string(k), 2),
            "  ", lpad(string(round(true_col[k]; digits=1)), 8),
            "  ", lpad(string(round(rec_col[k]; digits=1)), 8),
        )
    end
    println("  mean relative |ρ_rec − ρ_true|/ρ_true = ", round(100 * rel_err; digits=1), "%")
    println()
    println("PASS: synthetic pipeline recovers signal; loss fell ", round(drop_pct; digits=1), "%.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch err
        err isa InterruptException && rethrow()
        println(stderr, "Error: ", err)
        exit(1)
    end
end
