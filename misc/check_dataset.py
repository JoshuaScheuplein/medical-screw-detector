import json
import tifffile
from pathlib import Path

"""
Proejction Files:

Found 169 valid samples: ['Wrist04_1', 'Ankle05_2', 'Knee05_2', 'Wrist11_1', 'Leg01_2', 'Wrist10_2', 'Elbow04_2', 'Spine06_1', 'Ankle15_1', 'Wrist01_1', 'Knee05_3', 'Ankle10_3', 'Knee04_3', 'Ankle13_2', 'Foot01_1', 'Knee03_3', 'Elbow02_1', 'Wrist08_2', 'Knee02_3', 'Wrist13_2', 'Spine05_1', 'Ankle05_1', 'Ankle06_3', 'Elbow02_3', 'Ankle23_2', 'Ankle02_2', 'Spine02_3', 'Wrist05_2', 'Wrist12_3', 'Leg01_1', 'Ankle07_2', 'Wrist06_3', 'Ankle22_2', 'Wrist08_1', 'Ankle06_2', 'Wrist08_3', 'Ankle08_3', 'Ankle23_1', 'Ankle01_2', 'Ankle09_1', 'Spine04_1', 'Spine02_1', 'Ankle03_1', 'Elbow03_2', 'Ankle21_2', 'Ankle11_2', 'Wrist11_3', 'Leg01_3', 'Ankle19_1', 'Spine07_1', 'Ankle14_2', 'Knee08_1', 'Ankle08_2', 'Ankle10_2', 'Ankle15_3', 'Knee07_1', 'Wrist03_3', 'Spine06_3', 'Knee06_1', 'Ankle22_3', 'Knee05_1', 'Spine07_3', 'Ankle12_1', 'Spine03_1', 'Foot01_2', 'Knee04_2', 'Knee03_2', 'Wrist02_2', 'Ankle02_1', 'Knee02_2', 'Wrist07_2', 'Knee01_2', 'Ankle13_3', 'Ankle21_3', 'Ankle09_3', 'Ankle07_1', 'Spine01_1', 'Ankle01_3', 'Ankle18_3', 'Knee01_3', 'Wrist04_2', 'Elbow01_1', 'Ankle02_3', 'Wrist09_1', 'Wrist03_2', 'Elbow03_3', 'Wrist02_1', 'Ankle11_1', 'Ankle19_3', 'Elbow02_2', 'Elbow01_2', 'Wrist06_2', 'Ankle12_3', 'Wrist09_2', 'Ankle08_1', 'Spine04_2', 'Ankle03_3', 'Spine05_2', 'Ankle18_1', 'Wrist12_1', 'Wrist10_3', 'Spine03_3', 'Ankle04_1', 'Knee09_3', 'Spine01_2', 'Knee08_3', 'Wrist06_1', 'Knee07_3', 'Wrist05_3', 'Elbow03_1', 'Ankle14_3', 'Wrist05_1', 'Ankle19_2', 'Ankle17_2', 'Spine06_2', 'Knee06_3', 'Ankle21_1', 'Ankle17_3', 'Ankle23_3', 'Wrist11_2', 'Ankle05_3', 'Ankle20_1', 'Knee04_1', 'Knee03_1', 'Elbow01_3', 'Spine04_3', 'Wrist09_3', 'Knee02_1', 'Wrist07_1', 'Knee01_1', 'Ankle18_2', 'Wrist10_1', 'Spine03_2', 'Ankle14_1', 'Wrist01_2', 'Wrist03_1', 'Wrist01_3', 'Ankle01_1', 'Spine07_2', 'Wrist02_3', 'Ankle06_1', 'Spine05_3', 'Ankle07_3', 'Spine01_3', 'Ankle09_2', 'Ankle03_2', 'Elbow04_1', 'Ankle16_3', 'Wrist13_1', 'Ankle12_2', 'Ankle15_2', 'Knee09_1', 'Ankle10_1', 'Spine02_2', 'Ankle11_3', 'Wrist13_3', 'Wrist12_2', 'Ankle13_1', 'Ankle22_1', 'Ankle17_1', 'Knee09_2', 'Knee08_2', 'Ankle20_2', 'Ankle16_1', 'Ankle16_2', 'Knee07_2', 'Wrist04_3', 'Knee06_2', 'Wrist07_3']

Found 5 invalid samples: ['Foot01_3', 'Ankle04_2', 'Ankle20_3', 'Ankle04_3', 'Elbow04_3']
"""

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
