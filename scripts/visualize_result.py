import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import TABLEAU_COLORS

from utils.data_normalization import neglog_normalize

colors = list(TABLEAU_COLORS.values())

def main(image_dir, prediction_dir, same_color=False):

    datadir_b = image_dir / "V1-1to3objects-400projections-circular"
    datadir_p = prediction_dir / "V1-1to3objects-400projections-circular"
    detector_shape, pixel_size = (976, 976), 0.305

    # loop over folders and convert each set of tool volumes to landmarks
    samples = [f for f in os.listdir(datadir_p) if os.path.isdir(datadir_p / f)]
    print(f"processing {len(samples)} samples:")

    for sample in samples:
        folder_b = datadir_b / sample
        folder_p = datadir_p / sample

        # debug plot
        # if (folder_b / "projections.tiff").exists() and (folder_p / "predictions_test_48.json").exists(): # Original code
        if (folder_b / "projections.tiff").exists(): # Adapted code
            result_dir = folder_p / "results"
            result_dir.mkdir(parents=False, exist_ok=True)

            # read projections
            print(f"generating previews for {folder_b}")
            views = tifffile.imread(str(folder_b / "projections.tiff")).astype(np.float32)

            epochs = [f for f in folder_p.iterdir() if f.name.startswith("predictions")]

            for epoch in epochs:

                # Split the file name and extension
                epoch_num = epoch.stem.split("_")[-1]

                predictions = json.load(open(folder_p / epoch))
                for key, value in predictions["landmarks2d"].items():
                    j = int(key[5:])

                    fig = plt.figure(figsize=(12, 12))

                    view = predictions['landmarks2d'][f'view_{j}']
                    rotation = view["rotation"]
                    x, y, h, w = view["crop_region"]
                    v_flip = view["v_flip"]
                    h_flip = view["h_flip"]

                    img = views[j]
                    if v_flip:
                        img = np.flipud(img)
                    if h_flip:
                        img = np.fliplr(img)
                    img = np.rot90(img, k=rotation // 90)
                    img = img[x:x + w, y:y + h]

                    img = neglog_normalize(img)

                    plt.imshow(img, cmap='gray')

                    for i, screw in enumerate(view['predictions'].values()):
                        if screw["screw_prob"] < 0:
                            continue

                        head_x = screw['head'][0] * w
                        head_y = screw['head'][1] * h

                        tip_x = screw['tip'][0] * w
                        tip_y = screw['tip'][1] * h

                        color = 'red' if same_color else colors[i % len(colors)]
                        plot_screw(head_x, head_y, tip_x, tip_y, color=color)

                    # for i, screw in enumerate(predictions['landmarks2d'][f'view_{j}']['targets'].values()):
                    #     head_x = screw['head'][0] * w
                    #     head_y = screw['head'][1] * h
                    #
                    #     tip_x = screw['tip'][0] * w
                    #     tip_y = screw['tip'][1] * h
                    #
                    #     color = 'red' if same_color else colors[i % len(colors)]
                    #     plot_screw(head_x, head_y, tip_x, tip_y, color=color)

                    plt.tight_layout()
                    fig.savefig(result_dir / f'prediction_preview_view_{j}_epoch_{epoch_num}.png')
                    plt.close(fig)

        print("done\n")


def plot_screw(p0_x, p0_y, p1_x, p1_y, color):
    plt.plot([p0_x], [p0_y], 'o', c=color,  markersize=15, markeredgewidth=4, alpha=0.7)
    plt.plot([p1_x], [p1_y], 'x', c=color,  markersize=15, markeredgewidth=4, alpha=0.7)
    plt.plot([p0_x, p1_x], [p0_y, p1_y], '--', c=color, linewidth=4, alpha=0.7)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize predictions')

    if os.name == 'nt':
        image_dir_default = r"C:\Users\wagne\Desktop"
        prediction_dir_default = r"C:\Users\wagne\Desktop"
    else:
        # image_dir_default = r"/home/vault/iwi5/iwi5163h"
        # prediction_dir_default = r"/home/hpc/iwi5/iwi5163h/eff_detr_2024_06_10_02_52"
        image_dir_default = r"/home/vault/iwi5/iwi5165h"
        prediction_dir_default = r"/home/hpc/iwi5/iwi5165h/Screw-Detection-Results/Job-xxxxxx"

    parser.add_argument('--image_dir', type=str, default=image_dir_default,
                        help='Directory containing the images')
    parser.add_argument('--prediction_dir', type=str, default=prediction_dir_default,
                        help='Directory containing the predictions')

    args = parser.parse_args()

    main(Path(args.image_dir), Path(args.prediction_dir))
