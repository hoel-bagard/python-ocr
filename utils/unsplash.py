"""Utils script for the (Lite) Unsplash dataset.

The Lite Unsplash dataset is free for commercial use. It can be downloaded there: https://unsplash.com/data/lite/latest)
The documentation for the data format is there: https://github.com/unsplash/datasets/blob/master/DOCS.md
"""
from argparse import ArgumentParser
from pathlib import Path
from shutil import get_terminal_size
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pandas as pd


def show_img(img: np.ndarray, window_name: str = "Image"):
    """Displays an image until the user presses the "q" key.

    Args:
        img: The image that is to be displayed.
        window_name (str): The name of the window in which the image will be displayed.
    """
    while True:
        # Make the image full screen if it's above a given size (assume the screen isn't too small^^)
        if any(img.shape[:2] > np.asarray([1080, 1440])):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    parser = ArgumentParser(description="Download script for the unsplash dataset")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("--show_images", "-s", action="store_true", help="Display images.")
    parser.add_argument("--output_path", "-o", type=Path, default=None, help="If given, images will be saved there.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path
    show_imgs: bool = args.show_images

    assert output_path or show_imgs, "Use at least one of '-s', '-o' or this script doesn't do anything"

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    filename = data_path / "photos.tsv000"
    df = pd.read_csv(filename, sep='\t', header=0)

    for i in range(nb_imgs := len(df["photo_image_url"])):
        msg = f"Processing image {i+1}/{nb_imgs}  (decription : {df['photo_description'][i]})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        img_url = df["photo_image_url"][i] + "?fm=jpg&w=1080&q=85&fit=max"
        req = urlopen(Request(img_url, headers={'User-Agent': 'Mozilla/5.0'}))
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        if output_path:
            cv2.imwrite(str(output_path / f"./out_{i}.jpg"), img)
        if show_imgs:
            show_img(img)

    print("\nFinished processing the dataset.")


if __name__ == "__main__":
    main()
