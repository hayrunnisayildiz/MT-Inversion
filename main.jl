#!/usr/bin/env julia
# MT SciML Inversion — end-to-end CLI entry point
#
#   julia --project=. main.jl
#   julia --project=. main.jl --iterations 50 --lr 1.0
#   julia --project=. main.jl --data data/raw/USMTArray.edi

include(joinpath(@__DIR__, "src", "SciMLInversion.jl"))
using .SciMLInversion
using .SciMLInversion: DataIngestion, PriorModels, ForwardModel

const DEFAULT_DATA = joinpath(@__DIR__, "data", "raw", "USMTArray.edi")

function usage()
    println(
        """
        MT SciML Inversion

        Usage:
          julia --project=. main.jl [options]

        Options:
          --data PATH        Station data file (default: data/raw/USMTArray.edi)
          --iterations N     Gradient-descent steps (default: 20)
          --lr FLOAT         Learning rate (default: 0.5)
          --lambda FLOAT     Tikhonov weight (default: 1e-3)
          --rho FLOAT        Uniform prior resistivity Ω·m (default: 100)
          -h, --help         Show this help
        """,
    )
end

"""
Parse CLI flags into a NamedTuple. Unknown flags or bad values throw.
"""
function parse_args(args::Vector{String})
    opts = Dict{Symbol, Any}(
        :data => DEFAULT_DATA,
        :iterations => 20,
        :lr => 0.5,
        :lambda => 1e-3,
        :rho => 100.0,
        :help => false,
    )

    i = 1
    while i <= length(args)
        flag = args[i]
        if flag in ("-h", "--help")
            opts[:help] = true
            i += 1
        elseif flag in ("--data", "--edi") && i < length(args)
            opts[:data] = abspath(args[i + 1])
            i += 2
        elseif flag == "--iterations" && i < length(args)
            opts[:iterations] = parse(Int, args[i + 1])
            i += 2
        elseif flag == "--lr" && i < length(args)
            opts[:lr] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--lambda" && i < length(args)
            opts[:lambda] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--rho" && i < length(args)
            opts[:rho] = parse(Float64, args[i + 1])
            i += 2
        else
            error("Unknown or incomplete argument: $flag\nRun with --help for usage.")
        end
    end

    return (;
        data = String(opts[:data]),
        iterations = Int(opts[:iterations]),
        lr = Float64(opts[:lr]),
        lambda = Float64(opts[:lambda]),
        rho = Float64(opts[:rho]),
        help = Bool(opts[:help]),
    )
end

"""
Run the full pipeline: DataIngestion → PriorModels → ForwardModel → SciMLInversion.
"""
function run_pipeline(opts)
    println("MT SciML Inversion")
    println("  data       : ", opts.data)
    println("  iterations : ", opts.iterations)
    println("  lr         : ", opts.lr)
    println("  lambda     : ", opts.lambda)
    println("  prior ρ    : ", opts.rho, " Ω·m")
    println()

    # 1. DataIngestion — load station MT data
    println("── [1/4] DataIngestion ──")
    station_data = DataIngestion.load_raw_data(opts.data)
    app_res = DataIngestion.calculate_apparent_resistivity(station_data)
    println("Station : ", station_data.station_id)
    println("Freqs   : ", length(station_data.frequencies))
    println(
        "ρ_a range: ",
        round(minimum(app_res.rho_a), digits=2),
        " – ",
        round(maximum(app_res.rho_a), digits=2),
        " Ω·m",
    )
    println()

    # 2. PriorModels — build 3D prior resistivity model
    println("── [2/4] PriorModels ──")
    prior_model = PriorModels.create_3d_prior_model(default_rho=opts.rho)
    PriorModels.summarize_prior_model(prior_model)
    println()

    # 3. ForwardModel — initial 1D Wait response at prior
    println("── [3/4] ForwardModel ──")
    Z_pred = ForwardModel.compute_station_responses(prior_model, station_data.frequencies)
    println("Predicted impedances: ", length(Z_pred), " frequencies (grid-center column)")
    for i in 1:min(3, length(station_data.frequencies))
        f = station_data.frequencies[i]
        z_xy = Z_pred[i][1, 2]
        println(
            "  f = ",
            lpad(string(f), 12),
            " Hz  |Z_xy| = ",
            round(abs(z_xy), digits=4),
        )
    end
    println()

    # 4. SciMLInversion — Zygote gradient-descent inversion
    println("── [4/4] SciMLInversion ──")
    ps = run_inversion(
        station_data,
        prior_model;
        iterations = opts.iterations,
        lr = opts.lr,
        lambda = opts.lambda,
    )

    return (; station_data, prior_model, Z_pred, ps)
end

function main(args::Vector{String} = ARGS)
    opts = parse_args(args)
    if opts.help
        usage()
        return 0
    end

    isfile(opts.data) || error("Data file not found: $(opts.data)")
    run_pipeline(opts)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch err
        err isa InterruptException && rethrow()
        println(stderr, "Error: ", err)
        exit(1)
    end
end
