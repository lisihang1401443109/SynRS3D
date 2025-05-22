"""
classify_gsd.py  -  Put image filenames into g005 / g05 / g1 lists
-----------------------------------------------------------------
Run from the root that contains your dataset folders, e.g.:

    python classify_gsd.py /mnt/synrs3d/SynRS3D/data

"""

import sys
from pathlib import Path
from collections import defaultdict

import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
from tqdm import tqdm

def ensure_tif(name:str):
    low = name.lower()

    if low.endswith('.tif') or low.endswith('.tiff'):
        return name
    return name + '.tif'

#returns bucket to go to depending on gsd
def gsd_bucket(px_size):
    if 0.05 <= px_size < 0.30:
        return "g005"
    if 0.30 <= px_size < 0.60:
        return "g05"
    if 0.60 <= px_size <= 1.00:
        return "g1"
    return None

def pixel_size(tif_path):
    """Return |pixel width| in metres, or None if not georeferenced."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=NotGeoreferencedWarning)
        with rasterio.open(tif_path) as src:
            # If there's no transform (1-pixel grid) bail out
            if src.transform.is_identity:
                return None
            # 1) fully georeferenced & projected → use it
            if src.crs and src.crs.is_projected:
                return abs(src.transform.a)
            # 2) NO crs, but transform present → assume metres
            if src.crs is None:
                return abs(src.transform.a)
            # 3) geographic CRS (degrees) → caller would need to convert
            return None
        
def process_dataset(ds_path: Path):
    test_file = ds_path / "test.txt"
    if not test_file.exists():
        return {}
    
    #prepare output lists
    out_paths = {b: test_file.with_name(f"{b}.txt") for b in ("g005","g05","g1")}
    buckets = defaultdict(list)

    with test_file.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]
        print('The Number of filenames in test.txt',len(lines))

    for rel_path in tqdm(lines,desc=ds_path.name,unit='img'):
        rel_with_ext = ensure_tif(rel_path)
        tif_path = ds_path / 'opt' / rel_with_ext
        if not tif_path.exists():
            print(f"⚠ Missing file: {tif_path}", file=sys.stderr)
            continue
        px = pixel_size(tif_path)

        if px is None:
            print(f"⚠ No georef: {tif_path}", file=sys.stderr)
            continue
        bucket = gsd_bucket(px)

        if bucket:
            buckets[bucket].append(rel_path)
        else:
            print(f"⚠ GSD {px:.3f} m out of range: {tif_path}", file=sys.stderr)
        
    for b, lst in buckets.items():
        with out_paths[b].open("w") as f:
            f.write("\n".join(lst))
    return {k: len(v) for k,v in buckets.items()}


if __name__ == "__main__":
    datasets_paths = ['/mnt/synrs3d/SynRS3D/data/DFC19_OMA/',
                      '/mnt/synrs3d/SynRS3D/data/DFC19_JAX/',
                      '/mnt/synrs3d/SynRS3D/data/DFC18/',
                      '/mnt/synrs3d/SynRS3D/data/geonrw_rural/',
                      '/mnt/synrs3d/SynRS3D/data/geonrw_urban/',
                      '/mnt/synrs3d/SynRS3D/data/OGC_ARG/',
                      '/mnt/synrs3d/SynRS3D/data/OGC_ATL/'
                      ]
    
    total = defaultdict(int)
    for ds in sorted(datasets_paths):
        ds_path = Path(ds)
        if not ds_path.is_dir():
            continue
        counts = process_dataset(ds_path)
        if counts:
            print(f"{ds_path.name}: {counts}")
            for k,v in counts.items():
                total[k] += v

    print("\n=== overall totals ===")
    for k in ("g005", "g05", "g1"):
        print(f"{k:>4}: {total.get(k, 0)}")