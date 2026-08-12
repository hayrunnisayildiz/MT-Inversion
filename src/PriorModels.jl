module PriorModels

using Statistics: mean

"""
Geological / geophysical prior models and constraints for 3D MT inversion.
"""

export PriorModel3D, create_3d_prior_model, summarize_prior_model

"""
    PriorModel3D

3D prior resistivity model with grid spacing vectors and physical bounds.
"""
struct PriorModel3D
    nx::Int
    ny::Int
    nz::Int
    dx::Vector{Float64}
    dy::Vector{Float64}
    dz::Vector{Float64}
    resistivity::Array{Float64, 3}
    rho_min::Float64
    rho_max::Float64
end

"""
    create_3d_prior_model(; kwargs...)

Build a uniform horizontal grid with logarithmic vertical spacing (`dz` grows by
`z_factor` with depth) and a constant default resistivity fill.
"""
function create_3d_prior_model(;
    nx::Int = 20,
    ny::Int = 20,
    nz::Int = 15,
    dx_base::Float64 = 1000.0,
    dy_base::Float64 = 1000.0,
    dz_base::Float64 = 200.0,
    z_factor::Float64 = 1.15,
    default_rho::Float64 = 100.0,
)
    dx = fill(dx_base, nx)
    dy = fill(dy_base, ny)

    dz = Vector{Float64}(undef, nz)
    dz[1] = dz_base
    for k in 2:nz
        dz[k] = dz[k - 1] * z_factor
    end

    resistivity = fill(default_rho, nx, ny, nz)

    return PriorModel3D(
        nx, ny, nz,
        dx, dy, dz,
        resistivity,
        1.0, 10_000.0,
    )
end

"""
    summarize_prior_model(model::PriorModel3D)

Print grid extent, cell count, depth coverage, and resistivity statistics.
"""
function summarize_prior_model(model::PriorModel3D)
    rho = model.resistivity
    n_cells = model.nx * model.ny * model.nz
    total_depth = sum(model.dz)
    extent_x = sum(model.dx)
    extent_y = sum(model.dy)

    println("=== PriorModel3D Summary ===")
    println("Grid size: $(model.nx) × $(model.ny) × $(model.nz)")
    println("Total cells: $n_cells")
    println("Horizontal extent: X = $(round(extent_x, digits=2)) m, Y = $(round(extent_y, digits=2)) m")
    println("Total depth coverage: $(round(total_depth, digits=2)) m")
    println("dz range: $(round(model.dz[1], digits=2)) – $(round(model.dz[end], digits=2)) m")
    println(
        "Resistivity (Ω·m): min = $(round(minimum(rho), digits=4)), ",
        "max = $(round(maximum(rho), digits=4)), ",
        "mean = $(round(mean(rho), digits=4))",
    )
    println("Bounds: rho_min = $(model.rho_min), rho_max = $(model.rho_max)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = create_3d_prior_model()
    summarize_prior_model(model)
end

end # module
