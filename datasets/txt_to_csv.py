import os
import pandas as pd

def txt_to_df(path):
    df = pd.read_csv(path, sep=r"\s+")
    return df.iloc[:, 2:158:2]

def convert_all_txt(convert_limit = 1000):
    script_dir = os.path.dirname(__file__)
    input_dir = os.path.join(script_dir, "txt_files")
    output_dir = os.path.join(script_dir, "csv_files")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    convert_counter = 0
    for fname in os.listdir(input_dir):
        if convert_counter >= convert_limit:
            print(f"Convert limit ({convert_limit}) reached. Halting further conversions")
            break
        if not fname.lower().endswith(".txt"):
            continue
        in_path = os.path.join(input_dir, fname)
        if not os.path.isfile(in_path):
            continue
        try:
            df = txt_to_df(in_path)
            out_name = os.path.splitext(fname)[0] + ".csv"
            out_path = os.path.join(output_dir, out_name)
            df.to_csv(out_path, index=False)
            print(f"Converted {fname} -> {out_name}")
            convert_counter += 1
        except Exception as e:
            print(f"Failed to convert {fname}: {e}")

if __name__ == "__main__":
    convert_all_txt()