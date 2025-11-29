import os

def remove_tif_extension(root_dir):
    print(f"Searching for subset_train.txt files in {root_dir}...")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "OEM" not in dirpath:
            continue
        if "subset_train.txt" in filenames:
            file_path = os.path.join(dirpath, "subset_train.txt")
            print(f"Processing {file_path}...")
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = [line.replace('.tif', '') for line in lines]
                
                if new_lines != lines:
                    with open(file_path, 'w') as f:
                        f.writelines(new_lines)
                    print(f"  Updated {file_path}")
                else:
                    print(f"  No changes needed for {file_path}")
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Start searching from the parent directory of the current script (repo root)
    # Assuming script is in autoalbument/
    root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # Or specifically target the SynRS3D root if we are in autoalbument
    if os.path.basename(os.getcwd()) == "autoalbument":
         root = os.path.abspath("..")
    
    print(f"Root directory: {root}")
    remove_tif_extension(root)
