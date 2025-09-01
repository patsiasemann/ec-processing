import os
import shutil

# Folder where your CSV files are
input_folder = "H:\_SILVEX II 2025\Data\EC data\Silvia 2 (oben)\PEDDY\input\\"
output_folder = "H:\_SILVEX II 2025\Data\EC data\Silvia 2 (oben)\PEDDY\input\\"

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        csv_path = os.path.join(input_folder, filename)
        dat_filename = os.path.splitext(filename)[0] + ".dat"
        dat_path = os.path.join(output_folder, dat_filename)
        
        # Copy content unchanged
        shutil.copy(csv_path, dat_path)

print("Conversion finished!")