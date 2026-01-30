## Ensure we use the local project (important if script is run via `julia run_processing.jl`)
import Pkg
Pkg.activate(@__DIR__)

# Force local source preference on LOAD_PATH before resolving
local_src_dir = joinpath(@__DIR__, "src")
if !(local_src_dir in LOAD_PATH)
    pushfirst!(LOAD_PATH, local_src_dir)
end

# Try to resolve; if still remote, force include
local_src_file = joinpath(local_src_dir, "PEDDY.jl")
found_pkg = Base.find_package("PEDDY")
if found_pkg != local_src_file
    @info "Forcing local PEDDY load via include" found_pkg local_src_file
    # Blind include defines module PEDDY in Main; to avoid collision, only do if not already loaded
    if isdefined(Main, :PEDDY)
        @warn "PEDDY already defined but not from local path" current=pathof(PEDDY)
    else
        include(local_src_file)
    end
else
    using PEDDY
end

# If we had to include manually, ensure var exported (names check). If using, it's already in Main.
if !(@isdefined PEDDY)
    error("Failed to load local PEDDY")
end
if !( :MakeContinuous in names(PEDDY) )
    @warn "MakeContinuous still missing after local force. Consider: rm installed registry version (Pkg.rm(\"PEDDY\")) then Pkg.develop(path=@__DIR__)."
end
using Glob, Dates, Statistics
using DimensionalData

ped_path = try
    pathof(PEDDY)
catch
    "unknown"
end
@info "Using project" active_project=Base.active_project() pedddy_path=ped_path has_make=:MakeContinuous in names(PEDDY)

logger = PEDDY.ProcessingLogger()

# Sensor setup
sensor = PEDDY.CSAT3B()
needed_cols = collect(PEDDY.needs_data_cols(sensor))

# Data setup

# SLF PC
input_dir = raw"F:\Data\SILVEX-I\EC Data\Silvia3_Mitte\PEDDY\input"
#raw"H:\_SILVEX II 2025\Data\EC data\Silvia 2 (oben)\PEDDY\input\\"
output_dir = raw"F:\Data\SILVEX-I\EC Data\Silvia3_Mitte\PEDDY\output\2m"
#raw"H:\_SILVEX II 2025\Data\EC data\Silvia 2 (oben)\PEDDY\output\3m\\"

# Mac
# input_dir = raw"/Volumes/Expansion/Data/SILVEX II/Silvia 2 (oben)/PEDDY/input"
# output_dir = raw"/Volumes/Expansion/Data/SILVEX II/Silvia 2 (oben)/PEDDY/output"

input_files = "SILVEXI_Silvia3_sonics_*_2m.dat"
output_files = "_SILVEXI_Silvia3_2m.dat"

# Set up pipeline components
output = PEDDY.MemoryOutput()

wind_group = PEDDY.VariableGroup("Wind Components", [:Ux, :Uy, :Uz, :Ts], spike_threshold = 6.0)
gas_group  = PEDDY.VariableGroup("Gas Analyzer", [:H2O, :CO2], spike_threshold = 6.0)

despiking = PEDDY.SimpleSigmundDespiking(window_minutes = 5.0,
                                        variable_groups = [wind_group])#, gas_group])

gap_filling = PEDDY.GeneralInterpolation(; max_gap_size = 20,
                                        variables = needed_cols,
                                        method = PEDDY.Linear())

pipeline = PEDDY.EddyPipeline(; sensor = sensor,
                            quality_control = PEDDY.PhysicsBoundsCheck(),
                            despiking = despiking,
                            make_continuous = PEDDY.MakeContinuous(; step_size_ms = 50, max_gap_minutes= 5.0),
                            gap_filling = gap_filling,
                            gas_analyzer = nothing,
                            double_rotation = nothing,
                            output = output)

# Input options
fo = PEDDY.FileOptions(
    header = 1,
    delimiter = ",",
    comment = "#",
    timestamp_column = :TIMESTAMP,
    time_format = DateFormat("yyyy-mm-dd HH:MM:SS.s")
)

input = PEDDY.DotDatDirectory(
    directory = input_dir,
    high_frequency_file_glob = input_files,
    high_frequency_file_options = fo,
    low_frequency_file_glob = nothing,
    low_frequency_file_options = nothing,
)

matching_inputs = sort(Glob.glob(input_files, input_dir))
if isempty(matching_inputs)
    error("No input files matched pattern $(input_files) under $(input_dir).")
end
println("Processing $(length(matching_inputs)) input file(s):")
for (idx, file_path) in enumerate(matching_inputs)
    println("  [", idx, "/", length(matching_inputs), "] ", file_path)
end

# Helpers
format_ts(t) = t isa Dates.AbstractDateTime ? Dates.format(t, dateformat"yyyy-mm-ddTHH:MM:SS.s") : string(t)

"""
Count errors in data by variable:
 - diagnostic flags: count values > 0 (or > 63 for CSAT3)
 - other vars: count NaNs
"""
function count_errors(data)
    vars = collect(PEDDY.dims(data, PEDDY.Var))
    counts = Dict{Symbol, Int}()
    for v in vars
        col = parent(data[Var=At(v)])
        if v === :diag_sonic || v === :diag_gas
            if sensor === PEDDY.CSAT3()
                counts[v] = count(x -> (x > 63) && !(x isa AbstractFloat && isnan(x)), col)
            else    
                counts[v] = count(x -> (x > 0) && !(x isa AbstractFloat && isnan(x)), col)
            end
        else
            counts[v] = count(isnan, col)
        end
    end
    return counts
end

"""
Write data from a DimArray to a .dat (CSV-like) file.
 - Timestamps are formatted using `format_ts`
 - NaNs are written as "NaN"
"""
function write_dimarray_dat(data, filepath; delim = ",")
    ddims = PEDDY.dims(data)
    ti = collect(ddims[findfirst(x -> x isa PEDDY.Ti, ddims)])
    vars = collect(ddims[findfirst(x -> x isa PEDDY.Var, ddims)])
    A = data.data  # parent matrix with rows=time, cols=vars
    open(filepath, "w") do io
        # header
        println(io, join(vcat(["timestamp"], String.(vars)), delim))
        # rows
        for i in eachindex(ti)
            row = Vector{String}(undef, length(vars) + 1)
            row[1] = format_ts(ti[i])
            @inbounds for j in 1:length(vars)
                val = A[i, j]
                row[j + 1] = isnan(val) ? "NaN" : string(val)
            end
            println(io, join(row, delim))
        end
    end
end

function write_chunked_dimarray_dat(data, output_dir, output_suffix;
                                   chunk_duration = Minute(30), delim = ",")
    ti_dim = PEDDY.dims(data, PEDDY.Ti)
    ti = collect(ti_dim)
    isempty(ti) && return String[]
    n = length(ti)
    chunk_paths = String[]
    start_idx = 1
    while start_idx <= n
        start_time = ti[start_idx]
        chunk_end_exclusive = start_time + chunk_duration
        end_idx = start_idx
        while end_idx <= n && ti[end_idx] < chunk_end_exclusive
            end_idx += 1
        end
        end_idx = min(end_idx - 1, n)
        end_idx = max(end_idx, start_idx)
        chunk = data[Ti=start_idx:end_idx]
        chunk_prefix = Dates.format(start_time, dateformat"yyyy-mm-dd_HHMM")
        chunk_filename = string(chunk_prefix, output_suffix)
        chunk_path = joinpath(output_dir, chunk_filename)
        write_dimarray_dat(chunk, chunk_path; delim=delim)
        push!(chunk_paths, chunk_path)
        start_idx = end_idx + 1
    end
    return chunk_paths
end

# Read data and keep copy of raw data
hf, lf = PEDDY.read_data(input, sensor)
hf_raw = deepcopy(hf)

# Run pipeline with logging
pipeline_runtime = @elapsed PEDDY.process!(pipeline, hf, lf; logger=logger)
PEDDY.record_stage_time!(logger, :run_processing_script, pipeline_runtime)

# Get results
processed_sonicdata, _ = PEDDY.get_results(output)

# QA prints
println("Data dimensions: ", size(hf))
println("Error counts (raw data): \nNaNs per var; values > 0 per diag. flag")
for (k, v) in count_errors(hf_raw)
    println(rpad(String(k), 10), ": ", v)
end
println("Error counts (processed data): \nNaNs per var; values > 0 per diag. flag")
for (k, v) in count_errors(processed_sonicdata)
    println(rpad(String(k), 10), ": ", v)
end

# Build processed filenames per 30-minute chunk
mkpath(output_dir)
chunk_paths = write_chunked_dimarray_dat(processed_sonicdata, output_dir, output_files;
                                         chunk_duration=Minute(30), delim=",")
for path in chunk_paths
    println("Wrote processed file ", path)
end

first_chunk_path = isempty(chunk_paths) ? joinpath(output_dir, "processing") : chunk_paths[1]
log_path = replace(first_chunk_path, ".dat" => "_processing_log.csv")
PEDDY.write_processing_log(logger, log_path)
println("Wrote processing log ", log_path)