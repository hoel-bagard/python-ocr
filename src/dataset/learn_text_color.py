"""Learn pairs of background / text colors from the IIIT5K word dataset using k-means."""
from pathlib import Path

import cv2
import numpy as np

from src.utils.misc import clean_print, show_img


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=("Script to compute all the colors pairs."
                                         "Run with 'python -m src.dataset.learn_text_color <path>'."))
    parser.add_argument("dataset_dir", type=Path, help="Path to the IIIT5K dataset folder.")
    parser.add_argument("--output_dir", "-o", type=Path, default=Path("data"), help="Path to the output folder.")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir
    output_dir: Path = args.output_dir
    debug: bool = args.debug

    img_paths_list = list(dataset_dir.rglob("*.png"))  # Get all of the images from test and train
    nb_imgs = len(img_paths_list)
    pairs = np.empty((nb_imgs, 2, 3), dtype=np.float32)  # Array to store all the color pairs (1 pair per image)
    for i, img_path in enumerate(img_paths_list, start=0):
        clean_print(f"Processing image {img_path.name}    ({i+1}/{nb_imgs})", end="\r")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Paper said Lab color space
        pixel_vals = img.reshape((-1, 3))
        pixel_vals = np.float32(pixel_vals)

        # k-means with k=2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
        _, labels, centers = cv2.kmeans(pixel_vals, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        pairs[i] = centers

        if debug:
            # Convert back into uint8, and make the quantized version of the original image.
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((img.shape))

            concat_imgs = cv2.hconcat([img, segmented_image])
            concat_imgs = cv2.cvtColor(concat_imgs, cv2.COLOR_LAB2BGR)
            show_img(concat_imgs, img_path.name)

    np.save(output_dir / "color_pairs.npy", pairs)
