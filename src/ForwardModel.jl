if !isdefined(@__MODULE__, :PriorModels)
    include(joinpath(@__DIR__, "PriorModels.jl"))
end
if !isdefined(@__MODULE__, :DataIngestion)
    include(joinpath(@__DIR__, "DataIngestion.jl"))
end

module ForwardModel

using ..PriorModels: PriorModel3D, create_3d_prior_model
using ..DataIngestion: MTData, parse_edi_file

"""
Magnetotelluric forward modeling (1D layered-earth Wait recursion).
"""

export solve_1d_mt_forward, compute_station_responses

const MU0 = 4π * 1e-7

"""
    solve_1d_mt_forward(f, resistivities, thicknesses) -> ComplexF64

Compute the surface MT impedance for a 1D layered resistivity profile at
frequency `f` using Wait's recursive impedance formula.

Layer `1` is the shallowest; layer `n` is a semi-infinite half-space.
`length(thicknesses)` must equal `length(resistivities) - 1`.
"""
function solve_1d_mt_forward(f::Real, resistivities::AbstractVector, thicknesses::AbstractVector)
    n = length(resistivities)
    n >= 1 || error("resistivities must contain at least one layer")
    length(thicknesses) == n - 1 ||
        error("length(thicknesses) must equal length(resistivities) - 1")

    ω = 2π * float(f)

    if n == 1
        ρ = resistivities[1]
        k = sqrt(im * ω * MU0 / ρ)
        return k * ρ
    end

    ρ_bottom = resistivities[end]
    k_bottom = sqrt(im * ω * MU0 / ρ_bottom)
    Z = k_bottom * ρ_bottom

    for i in (n - 1):-1:1
        ρ = resistivities[i]
        h = thicknesses[i]
        k = sqrt(im * ω * MU0 / ρ)
        tanh_kh = tanh(k * h)
        kρ = k * ρ
        Z = kρ * (Z + kρ * tanh_kh) / (kρ + Z * tanh_kh)
    end

    return Z
end

"""
    _center_column(model::PriorModel3D) -> (resistivities, thicknesses)

Extract the vertical 1D resistivity column at the grid center (X, Y).
"""
function _center_column(model::PriorModel3D)
    ix = (model.nx + 1) ÷ 2
    iy = (model.ny + 1) ÷ 2

    resistivities = [model.resistivity[ix, iy, k] for k in 1:model.nz]
    thicknesses = model.nz > 1 ? model.dz[1:(model.nz - 1)] : Float64[]

    return resistivities, thicknesses
end

"""
    compute_station_responses(prior_model, frequencies) -> Vector{Matrix{ComplexF64}}

Extract the 1D resistivity column at the model center and compute the predicted
impedance tensor `Z_pred` for each frequency.

For a 1D structure, off-diagonal components satisfy `Zxy = -Zyx`; `Zxx` and
`Zyy` are zero.
"""
function compute_station_responses(
    prior_model::PriorModel3D,
    frequencies::AbstractVector{<:Real},
)
    resistivities, thicknesses = _center_column(prior_model)
    Z_pred = Vector{Matrix{ComplexF64}}(undef, length(frequencies))

    for (i, f) in pairs(frequencies)
        z = solve_1d_mt_forward(f, resistivities, thicknesses)
        Z_pred[i] = [
            0.0 + 0.0im z
            -z 0.0 + 0.0im
        ]
    end

    return Z_pred
end

"""
    compute_station_responses(resistivity, dz, frequencies) -> Vector

Differentiable 1D MT forward from a full 3D resistivity cube: uses the vertical
column at the grid center (same convention as `_center_column`).
"""
function compute_station_responses(
    resistivity::AbstractArray{<:Real, 3},
    dz::AbstractVector{<:Real},
    frequencies::AbstractVector{<:Real},
)
    nx, ny, nz = size(resistivity)
    ix = (nx + 1) ÷ 2
    iy = (ny + 1) ÷ 2

    resistivities = [resistivity[ix, iy, k] for k in 1:nz]
    thicknesses = nz > 1 ? dz[1:(nz - 1)] : eltype(dz)[]

    return [
        begin
            z = solve_1d_mt_forward(f, resistivities, thicknesses)
            [
                zero(z) z
                -z zero(z)
            ]
        end for f in frequencies
    ]
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== ForwardModel Test ===")

    prior = create_3d_prior_model(default_rho=100.0)

    edi_path = joinpath(@__DIR__, "..", "data", "raw", "USMTArray.edi")
    data = parse_edi_file(edi_path)
    Z_pred = compute_station_responses(prior, data.frequencies)

    println("Prior resistivity: 100.0 Ω·m (uniform)")
    println("Station column: grid center (ix=$(prior.nx ÷ 2 + 1), iy=$(prior.ny ÷ 2 + 1))")
    println("Frequencies: $(length(data.frequencies)) points")
    println()
    println("Theoretical impedance Z (first 5 frequencies):")
    for i in 1:min(5, length(data.frequencies))
        f = data.frequencies[i]
        z_xy = Z_pred[i][1, 2]
        println(
            "  f = ",
            lpad(string(f), 12),
            " Hz  Z_xy = ",
            round(real(z_xy), digits=4),
            " + ",
            round(imag(z_xy), digits=4),
            "i  |Z_xy| = ",
            round(abs(z_xy), digits=4),
            "  arg = ",
            round(rad2deg(angle(z_xy)), digits=2),
            "°",
        )
    end
    println()
    println("Theoretical apparent resistivity ρ_a (Ω·m, first 5 frequencies):")
    for i in 1:min(5, length(data.frequencies))
        f = data.frequencies[i]
        z_xy = Z_pred[i][1, 2]
        ω = 2π * f
        ρ_a = abs2(z_xy) / (ω * MU0)
        println(
            "  f = ",
            lpad(string(f), 12),
            " Hz  ρ_a = ",
            round(ρ_a, digits=2),
            "  φ = ",
            round(rad2deg(angle(z_xy)), digits=2),
            "°",
        )
    end
end

end # module
