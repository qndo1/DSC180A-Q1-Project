import os
import pandas as pd
import numpy as np

def validate_mocap_df(df, filename):
    n_cols = df.shape[1]
    if n_cols == 0 or n_cols % 3 != 0:
        raise ValueError(f"Invalid shape: {n_cols} columns. Must be divisible by 3 (XYZ triplets).")
    return True

def txt_to_df(path):
    df = pd.read_csv(path, sep=r"\s+")
    
    # Remove NaNs
    df = df.dropna(axis=1, how='all')
    
    # Skip Frame/Time columns
    # If remainder is 1, skip 1 (Frame). If remainder is 2, skip 2 (Frame + Time).
    remainder = df.shape[1] % 3
    df_coords = df.iloc[:, remainder:].copy()
    
    # The original headers are shifted and broken. We replace them with generic tags.
    # This prevents the "Header mismatch" error and makes the CSV clean.
    num_markers = df_coords.shape[1] // 3
    new_headers = []
    for i in range(num_markers):
        # Create temporal headers: M1_X, M1_Y, M1_Z, M2_X...
        new_headers.extend([f"M{i+1}_X", f"M{i+1}_Y", f"M{i+1}_Z"])
        
    df_coords.columns = new_headers
    
    return df_coords

def convert_all_txt(convert_limit=1000):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "txt_files")
    output_dir = os.path.join(script_dir, "csv_files")
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    convert_counter = 0
    for fname in os.listdir(input_dir):
        if convert_counter >= convert_limit:
            break
            
        if not fname.lower().endswith(".txt"):
            continue
            
        in_path = os.path.join(input_dir, fname)
        
        try:
            df = txt_to_df(in_path)
            
            # Validate
            validate_mocap_df(df, fname)
            
            # Save
            out_name = os.path.splitext(fname)[0] + ".csv"
            out_path = os.path.join(output_dir, out_name)
            df.to_csv(out_path, index=False)
            
            print(f"Converted {fname} -> {out_name} (Markers: {df.shape[1]//3})")
            convert_counter += 1
            
        except Exception as e:
            print(f"FAILED {fname}: {e}")

if __name__ == "__main__":
    convert_all_txt()