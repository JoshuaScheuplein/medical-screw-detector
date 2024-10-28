import json
import tifffile
from pathlib import Path

data_dir = Path("/home/vault/iwi5/iwi5165h/TestDir/2024-04-Scheuplein-Screw-Detection")

valid_samples, invalid_samples = [], []
for sample in data_dir.iterdir():
    print(f"\nInspecting sample '{str(sample)}'")

    projections_path = data_dir / Path(sample) / Path("projections.tiff")
    if projections_path.is_file():
        projections = tifffile.imread(projections_path)
        print(f"Found projections file with shape: {projections.shape}")
        valid_samples.append(str(sample).split("/")[-1])
    else:
        print(f"ERROR: Could not find projections file!")
        invalid_samples.append(str(sample).split("/")[-1])

print(f"\nFound {len(valid_samples)} valid samples:", valid_samples)
print(f"\nFound {len(invalid_samples)} invalid samples:", invalid_samples)
