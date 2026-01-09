import os
import shutil

# Folder where your files are
input_folder = "D:\\SILVEX II 2025\\EC data\\Silvia 1 (unten)\\converted"
output_folder = "D:\\SILVEX II 2025\\EC data\\Silvia 1 (unten)\\converted"

def rename_toa5_files(folder_path: str) -> None:
    """Rename any file containing 'TOA5' to use 'SILVEXII' instead."""
    for filename in os.listdir(folder_path):
        if "TOA5" not in filename:
            continue

        new_filename = filename.replace("TOA5", "SILVEXII")
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_filename)

        if os.path.exists(dst):
            print(f"Skipping rename for {filename}: {new_filename} already exists")
            continue

        os.rename(src, dst)
        print(f"Renamed {filename} -> {new_filename}")

rename_toa5_files(input_folder)

def convert_csv_to_dat(input_folder: str, output_folder: str) -> None:
    """Convert all CSV files in the input folder to DAT files in the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_folder, filename)
            dat_filename = os.path.splitext(filename)[0] + ".dat"
            dat_path = os.path.join(output_folder, dat_filename)
        
            # Copy content unchanged
            shutil.copy(csv_path, dat_path)

    print("Conversion finished!")