module DataIngestion

"""
Load and preprocess raw MT data (.edi, .txt, .xml) into Julia-ready formats.
"""

export MTData, parse_edi_file, calculate_apparent_resistivity, load_raw_data, save_processed_data

const MU0 = 4π * 1e-7

struct MTData
    station_id::String
    latitude::Float64
    longitude::Float64
    elevation::Float64
    frequencies::Vector{Float64}
    Z::Vector{Matrix{ComplexF64}}
end

function _parse_dms(value::AbstractString)
    s = strip(value)
    sign = 1.0
    if startswith(s, "-")
        sign = -1.0
        s = s[2:end]
    end
    parts = split(s, ':')
    deg = parse(Float64, parts[1])
    min = parse(Float64, parts[2])
    sec = parse(Float64, parts[3])
    return sign * (deg + min / 60 + sec / 3600)
end

function _parse_header_value(line::AbstractString)
    if occursin('=', line)
        _, value = split(line, '=', limit=2)
        value = strip(value)
        if startswith(value, '"') && endswith(value, '"')
            return value[2:(end - 1)]
        end
        return value
    end
    return strip(line)
end

function _block_count(line::AbstractString)
    m = match(r"//(\d+)", line)
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function _parse_float_block(lines::Vector{String}, start_idx::Int, expected_count::Int)
    values = Float64[]
    i = start_idx
    while length(values) < expected_count && i <= length(lines)
        line = strip(lines[i])
        if startswith(line, ">") || startswith(line, ">=")
            break
        end
        isempty(line) && (i += 1; continue)
        for token in split(line)
            push!(values, parse(Float64, token))
        end
        i += 1
    end
    length(values) == expected_count ||
        error("Expected $expected_count values, got $(length(values))")
    return values, i
end

function _find_block(lines::Vector{String}, tag::AbstractString)
    for (idx, line) in enumerate(lines)
        stripped = strip(line)
        startswith(stripped, ">$tag") || continue
        count = _block_count(stripped)
        count === nothing && error("Missing block size for >$tag")
        return idx, count
    end
    error("EDI block >$tag not found")
end

function parse_edi_file(filepath::String)::MTData
    lines = readlines(filepath)

    station_id = ""
    latitude = 0.0
    longitude = 0.0
    elevation = 0.0

    for line in lines
        stripped = strip(line)
        if startswith(stripped, "DATAID=")
            station_id = _parse_header_value(stripped)
        elseif startswith(stripped, "LAT=")
            latitude = _parse_dms(_parse_header_value(stripped))
        elseif startswith(stripped, "LONG=")
            longitude = _parse_dms(_parse_header_value(stripped))
        elseif startswith(stripped, "ELEV=")
            elevation = parse(Float64, _parse_header_value(stripped))
        end
    end

    isempty(station_id) && error("DATAID not found in EDI file")

    freq_idx, nfreq = _find_block(lines, "FREQ")
    zxxr_idx, _ = _find_block(lines, "ZXXR")
    zxxi_idx, _ = _find_block(lines, "ZXXI")
    zxyr_idx, _ = _find_block(lines, "ZXYR")
    zxyi_idx, _ = _find_block(lines, "ZXYI")
    zyxr_idx, _ = _find_block(lines, "ZYXR")
    zyxi_idx, _ = _find_block(lines, "ZYXI")
    zyyr_idx, _ = _find_block(lines, "ZYYR")
    zyyi_idx, _ = _find_block(lines, "ZYYI")

    frequencies, _ = _parse_float_block(lines, freq_idx + 1, nfreq)
    zxxr, _ = _parse_float_block(lines, zxxr_idx + 1, nfreq)
    zxxi, _ = _parse_float_block(lines, zxxi_idx + 1, nfreq)
    zxyr, _ = _parse_float_block(lines, zxyr_idx + 1, nfreq)
    zxyi, _ = _parse_float_block(lines, zxyi_idx + 1, nfreq)
    zyxr, _ = _parse_float_block(lines, zyxr_idx + 1, nfreq)
    zyxi, _ = _parse_float_block(lines, zyxi_idx + 1, nfreq)
    zyyr, _ = _parse_float_block(lines, zyyr_idx + 1, nfreq)
    zyyi, _ = _parse_float_block(lines, zyyi_idx + 1, nfreq)

    Z = Vector{Matrix{ComplexF64}}(undef, nfreq)
    for i in 1:nfreq
        Z[i] = [
            complex(zxxr[i], zxxi[i]) complex(zxyr[i], zxyi[i])
            complex(zyxr[i], zyxi[i]) complex(zyyr[i], zyyi[i])
        ]
    end

    return MTData(station_id, latitude, longitude, elevation, frequencies, Z)
end

function calculate_apparent_resistivity(data::MTData)
    nfreq = length(data.frequencies)
    rho_a = Vector{Float64}(undef, nfreq)
    phi = Vector{Float64}(undef, nfreq)

    for i in 1:nfreq
        z_xy = data.Z[i][1, 2]
        omega = 2π * data.frequencies[i]
        rho_a[i] = abs2(z_xy) / (omega * MU0)
        phi[i] = rad2deg(angle(z_xy))
    end

    return (rho_a=rho_a, phi=phi)
end

function load_raw_data(path::AbstractString)
    if endswith(lowercase(path), ".edi")
        return parse_edi_file(String(path))
    end
    error("Not implemented: load_raw_data($path)")
end

function save_processed_data(data, path::AbstractString)
    error("Not implemented: save_processed_data(..., $path)")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .DataIngestion

    edi_path = joinpath(@__DIR__, "..", "data", "raw", "USMTArray.edi")
    data = parse_edi_file(edi_path)
    result = calculate_apparent_resistivity(data)

    println("=== MT Data Summary ===")
    println("Station ID : ", data.station_id)
    println("Latitude   : ", round(data.latitude, digits=6), "°")
    println("Longitude  : ", round(data.longitude, digits=6), "°")
    println("Elevation  : ", data.elevation, " m")
    println("Frequencies: ", length(data.frequencies), " points")
    println("Freq range : ", minimum(data.frequencies), " – ", maximum(data.frequencies), " Hz")
    println()
    println("Impedance Z_xy (first 5 frequencies):")
    for i in 1:min(5, length(data.frequencies))
        z_xy = data.Z[i][1, 2]
        println(
            "  f = ",
            lpad(string(data.frequencies[i]), 12),
            " Hz  |Z_xy| = ",
            round(abs(z_xy), digits=4),
            "  arg = ",
            round(rad2deg(angle(z_xy)), digits=2),
            "°",
        )
    end
    println()
    println("Apparent resistivity ρ_a (Ω·m) and phase φ (°):")
    for i in 1:min(5, length(data.frequencies))
        println(
            "  f = ",
            lpad(string(data.frequencies[i]), 12),
            " Hz  ρ_a = ",
            round(result.rho_a[i], digits=2),
            "  φ = ",
            round(result.phi[i], digits=2),
            "°",
        )
    end
    println("  ...")
    println(
        "  f = ",
        lpad(string(data.frequencies[end]), 12),
        " Hz  ρ_a = ",
        round(result.rho_a[end], digits=2),
        "  φ = ",
        round(result.phi[end], digits=2),
        "°",
    )
end
