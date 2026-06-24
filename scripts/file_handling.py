import os
import re
import shutil

# Folder where your files are
input_folder = "F:\\Data\\SILVEX-I\\EC Data\\Silvia3_Mitte\\converted"
output_folder = "F:\\Data\\SILVEX-I\\EC Data\\Silvia3_Mitte\\converted"

def rename_toa5_files(folder_path: str) -> None:
    """Rename any file containing 'TOA5' to use 'SILVEXI' instead."""
    for filename in os.listdir(folder_path):
        if "TOA5" in filename:
            new_filename = filename.replace("TOA5", "SILVEXI")

        if "CSAT" in new_filename:
            new_filename = new_filename.replace("CSAT", "")

        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_filename)
        
        if os.path.exists(dst):
            print(f"Skipping rename for {filename}: {new_filename} already exists")
            continue

        os.rename(src, dst)
        print(f"Renamed {filename} -> {new_filename}")


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


def pad_file_numbers(folder_path: str) -> None:
    """
    Rename files matching 'SILVEXI_Silvia2_sonics_*_2m.dat' pattern
    to pad the number * with leading zeros to 3 digits.
    
    Example: SILVEXI_Silvia2_sonics_5_2m.dat -> SILVEXI_Silvia2_sonics_005_2m.dat
    """
    # Pattern to match files like SILVEXI_Silvia2_sonics_5_2m.dat
    pattern = re.compile(r'^(SILVEXI_Silvia3_sonics_)(\d+)(_2m\.dat)$')
    
    renamed_count = 0
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            prefix, number, suffix = match.groups()
            # Pad the number to 3 digits
            padded_number = number.zfill(3)
            new_filename = f"{prefix}{padded_number}{suffix}"
            
            # Skip if already correctly formatted
            if filename == new_filename:
                continue
            
            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_filename)
            
            if os.path.exists(dst):
                print(f"Skipping {filename}: {new_filename} already exists")
                continue
            
            os.rename(src, dst)
            print(f"Renamed: {filename} -> {new_filename}")
            renamed_count += 1
    
    print(f"\nCompleted! Renamed {renamed_count} file(s).")


# Uncomment the line below to run the function
pad_file_numbers(r"F:\Data\SILVEX-I\EC Data\Silvia3_Mitte\PEDDY\input")