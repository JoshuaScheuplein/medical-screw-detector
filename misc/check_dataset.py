import jsono
import tifffile
from pathlib import Path

data_dir = Path("/home/vault/iwi5/iwi5165h/TestDir/2024-04-Scheuplein-Screw-Detection")

for sample in data_dir.iterdir():
    print(f"\nInspecting sample {sample}")

    projections_path = data_dir / Path(sample) / Path(projections.tiff)
    if projections_path.isfile():
        projections = tifffile.imread(projections_path)
        print(f"Found projections file with shape: {projections.shape}")
    else:
        print(f"ERROR: Could not find projections file!")
