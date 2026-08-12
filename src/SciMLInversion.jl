module SciMLInversion

"""
Scientific machine learning inversion pipeline for MT data.

Builds trainable Lux parameters from a 3D prior resistivity model, evaluates a
data-misfit + Tikhonov loss, and runs Zygote-based gradient descent.
"""

include(joinpath(@__DIR__, "DataIngestion.jl"))
include(joinpath(@__DIR__, "PriorModels.jl"))
include(joinpath(@__DIR__, "ForwardModel.jl"))

using .DataIngestion
using .PriorModels
using .ForwardModel

using Lux
using Random
using Zygote
using Optim
using ComponentArrays

export DataIngestion, ForwardModel, PriorModels
export build_inversion_model, compute_loss, loss_terms, run_inversion, ResistivityField

"""
    ResistivityField

Lux layer whose only trainable parameters are the log-resistivity field
`log_rho` on the 3D inversion grid.
"""
struct ResistivityField <: Lux.AbstractLuxLayer
    nx::Int
    ny::Int
    nz::Int
    log_rho_init::Array{Float64, 3}
end

function Lux.initialparameters(::AbstractRNG, layer::ResistivityField)
    return (log_rho = copy(layer.log_rho_init),)
end

function Lux.initialstates(::AbstractRNG, ::ResistivityField)
    return NamedTuple()
end

"""
    build_inversion_model(prior_model::PriorModel3D) -> ComponentArray

Create a Lux-compatible trainable parameter structure `ps` from the prior
resistivity cube. Parameters are stored as `log(ρ)` so positivity is preserved
under unconstrained updates.
"""
function build_inversion_model(prior_model::PriorModel3D)
    log_rho0 = log.(prior_model.resistivity)
    layer = ResistivityField(prior_model.nx, prior_model.ny, prior_model.nz, log_rho0)
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    ps, _st = Lux.setup(rng, layer)
    return ComponentArray(ps)
end

"""
    loss_terms(ps, Z_obs, frequencies, prior_model; lambda=1e-3) -> NamedTuple

Absolute impedance misfit plus Tikhonov regularization:

```
L_misfit = mean_f ‖Z_pred − Z_obs‖²
L_reg    = mean ‖ρ − ρ_prior‖²
L        = L_misfit + λ L_reg
```
"""
function loss_terms(ps, Z_obs, frequencies, prior_model::PriorModel3D; lambda=1e-3)
    ρ = exp.(ps.log_rho)
    Z_pred = compute_station_responses(ρ, prior_model.dz, frequencies)

    misfit = zero(eltype(ρ))
    @inbounds for i in eachindex(Z_obs, Z_pred)
        Δ = Z_pred[i] .- Z_obs[i]
        misfit = misfit + sum(abs2, Δ)
    end
    misfit = misfit / length(Z_obs)

    ρ_prior = prior_model.resistivity
    reg = sum(abs2, ρ .- ρ_prior) / length(ρ_prior)

    return (total = misfit + lambda * reg, misfit = misfit, reg = reg)
end

"""
    compute_loss(ps, Z_obs, frequencies, prior_model; lambda=1e-3) -> Real

Scalar loss used by Zygote (see [`loss_terms`](@ref)).
"""
function compute_loss(ps, Z_obs, frequencies, prior_model::PriorModel3D; lambda=1e-3)
    return loss_terms(ps, Z_obs, frequencies, prior_model; lambda=lambda).total
end

"""
    run_inversion(station_data, prior_model; iterations=20, lr=0.5, lambda=1e-3)

Gradient descent on `log ρ` with Zygote gradients. Prints total / misfit / reg
each iteration and returns the optimized parameters `ps`.
"""
function run_inversion(
    station_data::MTData,
    prior_model::PriorModel3D;
    iterations::Int = 20,
    lr::Float64 = 0.5,
    lambda::Float64 = 1e-3,
)
    ps = build_inversion_model(prior_model)
    Z_obs = station_data.Z
    frequencies = station_data.frequencies

    println("=== SciML MT Inversion ===")
    println("Station : ", station_data.station_id)
    println("Freqs   : ", length(frequencies))
    println("Grid    : ", prior_model.nx, "×", prior_model.ny, "×", prior_model.nz)
    println("Steps   : ", iterations)
    println("Method  : Zygote + gradient descent (lr=", lr, ")")
    println()

    for iter in 1:iterations
        terms = loss_terms(ps, Z_obs, frequencies, prior_model; lambda=lambda)
        _, back = Zygote.pullback(
            p -> compute_loss(p, Z_obs, frequencies, prior_model; lambda=lambda),
            ps,
        )
        ∇ps = back(one(terms.total))[1]
        ps = ps .- lr .* ∇ps

        println(
            "Iter ",
            lpad(string(iter), 2),
            "/",
            iterations,
            "  Loss = ",
            terms.total,
            "  (misfit = ",
            terms.misfit,
            ", reg = ",
            terms.reg,
            ")",
        )
    end

    final = loss_terms(ps, Z_obs, frequencies, prior_model; lambda=lambda)
    println()
    println(
        "Final Loss = ",
        final.total,
        "  (misfit = ",
        final.misfit,
        ", reg = ",
        final.reg,
        ")",
    )
    return ps
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .SciMLInversion
    using .SciMLInversion: DataIngestion, PriorModels

    edi_path = joinpath(@__DIR__, "..", "data", "raw", "USMTArray.edi")

    println("Loading station data from ", edi_path)
    station_data = DataIngestion.parse_edi_file(edi_path)
    prior_model = PriorModels.create_3d_prior_model(default_rho=100.0)

    SciMLInversion.run_inversion(station_data, prior_model; iterations=20)
end
