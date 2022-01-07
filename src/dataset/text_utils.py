import random
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.misc import clean_print


class BBox(NamedTuple):
    """Bounding box helper."""
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int


def sample_font(font_dir: Path, jp_only: bool = True) -> ImageFont:
    """Sample a random font from all the available ones.

    TODO: Add random kerning, oblique, curve, size, etc...
    TODO: See if the fonts  can be pre-loaded (to make it faster).

    Args:
        font_dir: Path to a dir with all the available fonts (can have subfolders).
        jp_only: If true then will filter out fonts that are not in a subfolder with "_JP" in its name.

    Returns:
        A random font.
    """
    exts = [".ttf", ".otf"]
    font_paths_list = list([p for p in font_dir.rglob('*') if p.suffix in exts and "_JP" in str(p.parent)])
    font = ImageFont.truetype(str(random.choice(font_paths_list)), 34)
    return font


def render_text(font: ImageFont,
                text: str,
                debug: bool = False) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """Render text using the given font. Also compute the text's mask and bounding boxes.

    Args:
        font: The font to use.
        text: The text to render.
        debug: If true, then draw the bounding boxes aroung the characters.

    Returns:
        An image with the rendered text, along with the corresponding mask and charater bounding boxes.
    """
    text_img = Image.new("RGB", (500, 100))
    draw = ImageDraw.Draw(text_img)
    xy = (0, 0)
    draw.text(xy, text, font=font)

    # Get the masks and bbox for the font
    x_offset = 0  # Tracks the place of successive characters.
    bboxes: list[tuple[int, int, int, int]] = []
    for char in text:
        text_mask = font.getmask(char)

        # See also: https://github.com/python-pillow/Pillow/issues/3921
        # TODO: adjust the direction
        # TODO: y_offset for multiple lines
        bbox = BBox(*font.getbbox(char, direction=None, features=None, language=None))
        adjusted_bbox = BBox(bbox[0]+x_offset, bbox[1], bbox[2]+x_offset, bbox[3])
        bboxes.append(adjusted_bbox)
        if debug:
            draw.rectangle(adjusted_bbox, fill=None, outline="red")
        x_offset += bbox.bottom_right_x - bbox.top_left_x

    text_img = np.asarray(text_img)
    text_img = cv2.cvtColor(text_img, cv2.COLOR_RGB2BGR)

    return text_img, text_mask, bboxes


def place_text(img: np.ndarray, font: ImageFont, text: str, debug: bool = False) -> np.ndarray:
    """Place text.

    Args:
        img: The image to use as the background.
        font: The font to use.
        text: The text to render.
        debug: If true, then draw the bounding boxes aroung the characters.

    Returns:
        ???.
    """
    # Do a dummy text region (maybe just check that the bb fits on the img)

    # original:
    # From the pregenerated region, filter the one that can be used.

    text_img, text_mask, bboxes = render_text(font, text, debug=debug)

    # # Poisson Image Editing
    # # Read images : src image will be cloned into dst
    # img = cv2.imread("images/wood-texture.jpg")
    # obj = cv2.imread("images/iloveyouticket.jpg")

    # # Create an all white mask
    # mask = 255 * np.ones(obj.shape, obj.dtype)

    # # The location of the center of the src in the dst
    # width, height, channels = im.shape
    # center = (height/2, width/2)

    # # Seamlessly clone src into dst and put the results in output
    # normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
    # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

    # # Write results
    # cv2.imwrite("images/opencv-normal-clone-example.jpg", normal_clone)
    # cv2.imwrite("images/opencv-mixed-clone-example.jpg", mixed_clone)

    return text_img
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
        img = place_text(img, font, "hello国", debug=True)

        while True:
            cv2.imshow(img_path.name, img)
            key = cv2.waitKey(10)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

        break
