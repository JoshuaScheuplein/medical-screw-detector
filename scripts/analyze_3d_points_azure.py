import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper script')

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
    
    parser.add_argument('--azure', type=str, default="no") # Additionally added

    args = parser.parse_args()

    os.system(f"echo 'args.image_dir {args.image_dir}'")
    os.system(f"echo 'args.prediction_dir {args.prediction_dir}'")
    os.system(f"echo 'args.azure {args.azure}'")

    if args.azure == "yes":
        os.system("echo '\n#################################################################'")
        os.system("echo 'Current PYTHONPATH:'")
        os.system("echo $PYTHONPATH")

        os.system("echo 'Current Directory:'")
        os.system("echo $(pwd)")

        os.system("export PYTHONPATH=$(pwd)")

        os.system("echo 'New PYTHONPATH:'")
        os.system("echo $PYTHONPATH")
        os.system("echo '#################################################################'")

    os.system(f"python scripts/analyze_3d_points.py --image_dir={args.image_dir} --prediction_dir={args.prediction_dir}")
