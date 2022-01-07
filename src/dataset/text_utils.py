import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from src.utils.misc import clean_print


def sample_font(font_dir: Path) -> ImageFont:
    exts = [".ttf", ".otf"]
    font_paths_list = list([p for p in font_dir.rglob('*') if p.suffix in exts])
    font = ImageFont.truetype(str(random.choice(font_paths_list)), 26)
    return font


def place_text(img: np.ndarray, font: ImageFont) -> np.ndarray:
    # Do a dummy text region (maybe just check that the bb fits on the img)

    # original:
    # From the pregenerated region, filter the one that can be used.

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 10), "hello国", font=font)
    img = np.asarray(img_pil)

    return img


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Text utils test script. Run with 'python -m src.dataset.text_utils <path>'.")
    parser.add_argument("img_dir", type=Path, help="Path to the folder with the background images")
    parser.add_argument("--font_dir", "-f", type=Path, default=Path("src/dataset/fonts"),
                        help="Path to the folder with the font files.")
    args = parser.parse_args()

    img_dir: Path = args.img_dir
    font_dir: Path = args.font_dir

    exts = [".jpg", ".png", ".bmp"]
    img_paths_list = list([p for p in img_dir.rglob('*') if p.suffix in exts])
    nb_imgs = len(img_paths_list)
    for i, img_path in enumerate(img_paths_list, start=1):
        clean_print(f"Processing image {img_path.name}    ({i}/{nb_imgs})", end="\r")

        img = cv2.imread(str(img_path))

        font = sample_font(font_dir)
        img = place_text(img, font)

        while True:
            cv2.imshow(img_path.name, img)
            key = cv2.waitKey(10)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
