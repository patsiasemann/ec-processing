import pandas as pd
import glob
from pathlib import Path

# Keep and rename these columns
columns_to_keep = [
    "time", "u", "v", "w", "T", #"co2", "h2o", 
    #"airtemperature", "airpressure", 
    "diagsonic"#, "diagirg"
]

rename_dict = {
    "time": "TIMESTAMP",
    "u": "Ux",
    "v": "Uy",
    "w": "Uz",
    "T": "Ts",
    "diagsonic": "diag_sonic",
    #"co2": "CO2",
    #"h2o": "H2O",
    #"diagirg": "diag_gas",
    #"airtemperature": "T",
    #"airpressure": "P"
}

# File path
datapath = "H:\_SILVEX II 2025\Data\EC data\Silvia 2 (oben)\converted\\"
outputpath = "H:\_SILVEX II 2025\Data\EC data\Silvia 2 (oben)\PEDDY\input\\"

outdir = Path(outputpath)
outdir.mkdir(parents=True, exist_ok=True)

files = glob.glob(datapath + "SILVEXII_Silvia2_sonics_*_3m.csv")

for file in files:
    filename = file.split("\\")[-1]
    print(f"\n---------- Processing file: {filename} ----------")
    # Skip first line, use line 1 as header
    df = pd.read_csv(file, encoding='utf-8')

    # Check what the column names actually are
    print("Column names found in file:")
    print(df.columns.tolist())

    # Apply selection and renaming
    df_selected = df[columns_to_keep].rename(columns=rename_dict)

    # Export to CSV
    out_path = (outdir / filename).with_suffix('.dat')
    df_selected.to_csv(out_path, index=False, na_rep="NaN")
    