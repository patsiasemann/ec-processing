using PEDDY, Glob, Dates

# Sensor setup
sensor = PEDDY.IRGASON()
needed_cols = collect(PEDDY.needs_data_cols(sensor))

# Set up pipeline components
output = PEDDY.MemoryOutput()
gap_filling = PEDDY.GeneralInterpolation(; max_gap_size = 20, 
                                        variables = needed_cols,
                                        method = PEDDY.Linear())

pipeline = PEDDY.EddyPipeline(; sensor = sensor,
                            quality_control = PEDDY.PhysicsBoundsCheck(),
                            despiking = PEDDY.SimpleSigmundDespiking(window_minutes = 10.0),
                            gap_filling = gap_filling,
                            gas_analyzer = nothing,
                            double_rotation = PEDDY.WindDoubleRotation(block_duration_minutes = 1.0),
                            mrd = PEDDY.OrthogonalMRD(M=14),
                            output = output
)

# Data setup
datapath = raw"H:\_SILVEX II 2025\Data\EC data\Silvia 2 (oben)\PEDDY\input\\"

fo = PEDDY.FileOptions(
    header = 1,
    delimiter = ",",
    comment = "#",
    timestamp_column = :TIMESTAMP,
    time_format = DateFormat("yyyy-mm-ddTHH:MM:SS.s")
)

input = PEDDY.DotDatDirectory(
    directory = datapath,
    high_frequency_file_glob = "SILVEXII_Silvia2_sonics_001_1m.dat",
    high_frequency_file_options = fo,
    low_frequency_file_glob = nothing,
    low_frequency_file_options = nothing
)

# Read data
hf, lf = PEDDY.read_data(input, sensor)
hf

# Run pipeline
PEDDY.process!(pipeline, hf, lf)

# Get results
processed_sonicdata, _ = PEDDY.get_results(output)
processed_sonicdata

mrd = PEDDY.OrthogonalMRD(M=14)
PEDDY.decompose!(mrd, hf, lf)
res = PEDDY.get_mrd_results(mrd)
res