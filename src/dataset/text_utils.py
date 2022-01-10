import random
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.misc import clean_print, show_img


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
    font_size = random.randint(40, 200)
    font = ImageFont.truetype(str(random.choice(font_paths_list)), font_size)
    return font


def render_text(font: ImageFont,
                text: str) -> tuple[np.ndarray, np.ndarray, list[BBox]]:
    """Render text using the given font. Also compute the text's mask and bounding boxes.

    Args:
        font: The font to use.
        text: The text to render.

    Returns:
        An image with the rendered text, along with the corresponding mask and charater bounding boxes.
    """
    size = font.getsize(text)
    text_img = Image.new("RGB", size, color=(0, 0, 0))
    draw = ImageDraw.Draw(text_img)
    random_color = tuple(np.random.choice(range(256), size=3))  # TODO: have the color depend on the background
    draw.text((0, 0), text, fill=random_color, font=font)
    text_img = np.asarray(text_img)
    text_img = cv2.cvtColor(text_img, cv2.COLOR_RGB2BGR)

    # Get the masks and bbox for the font
    x_offset = 0  # Tracks the place of successive characters.
    bboxes: list[BBox] = []
    for char in text:
        # See also: https://github.com/python-pillow/Pillow/issues/3921
        # TODO: adjust the direction
        # TODO: y_offset for multiple lines
        bbox = BBox(*font.getbbox(char, direction=None, features=None, language=None))
        adjusted_bbox = BBox(bbox[0]+x_offset, bbox[1], bbox[2]+x_offset, bbox[3])
        bboxes.append(adjusted_bbox)
        x_offset += bbox.bottom_right_x - bbox.top_left_x

    # Create a mask of the text. By just redrawing it in black and white.
    # If using the mask directly for the seamless cloning, the the mask's stroke width needs to be much bigger than
    # the actual stroke width fot the seamless cloning to work.
    text_mask = Image.new('1', size, color=0)
    draw = ImageDraw.Draw(text_mask)
    draw.text((0, 0), text, fill=1, font=font, stroke_width=0)
    text_mask = np.asarray(text_mask, dtype=np.uint8)*255

    return text_img, text_mask, bboxes


def place_text(img: np.ndarray, font: ImageFont, text: str, debug: bool = False) -> np.ndarray:
    """Place text.

    Args:
        img: The image to use as the background.
        font: The font to use.
        text: The text to render.
        debug: If true, then draw the bounding boxes aroung the characters.

    Returns:
        The image with the text on it.
    """
    # Do a dummy text region (maybe just check that the bb fits on the img)
    # original:
    # From the pregenerated region, filter the one that can be used.

    text_img, text_mask, bboxes = render_text(font, text)

    # Poisson Image Editing
    width, height, _ = img.shape
    text_center = (height//2, width//2)
    mask = np.full_like(text_mask, 255)
    merged_img = cv2.seamlessClone(text_img, img, mask, text_center, cv2.MIXED_CLONE)

    # The seamless cloning doesn't work really well: the background stays in part and creates artifacts (shadows)
    # Since we have an exact mask of the text, we can remove the artifacts.
    my_mask = np.asarray(np.where(text_mask != 255))
    my_mask += np.expand_dims(np.asarray(text_center)[::-1] - np.asarray(text_img.shape[:2])//2, axis=-1)
    my_mask = tuple(my_mask)
    merged_img[my_mask] = img[my_mask]
    img = merged_img

    # Draw the bounding boxes if needed
    if debug:
        char_bbox: BBox
        for char_bbox in bboxes:
            top_left = np.asarray(char_bbox[:2]) + np.asarray(text_center) - np.asarray(text_img.shape[:2][::-1])//2
            bottom_right = np.asarray(char_bbox[2:]) + np.asarray(text_center) - np.asarray(text_img.shape[:2][::-1])//2
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), thickness=2, lineType=cv2.LINE_4)

    return img


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Text utils test script. Run with 'python -m src.dataset.text_utils <path>'.")
    parser.add_argument("img_dir", type=Path, help="Path to the folder with the background images")
    parser.add_argument("--font_dir", "-f", type=Path, default=Path("src/dataset/fonts"),
                        help="Path to the folder with the font files.")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    args = parser.parse_args()

    img_dir: Path = args.img_dir
    font_dir: Path = args.font_dir
    debug: bool = args.debug

    exts = [".jpg", ".png", ".bmp"]
    img_paths_list = list([p for p in img_dir.rglob('*') if p.suffix in exts])
    nb_imgs = len(img_paths_list)
    for i, img_path in enumerate(img_paths_list, start=1):
        clean_print(f"Processing image {img_path.name}    ({i}/{nb_imgs})", end="\r")

        img = cv2.imread(str(img_path))

        font = sample_font(font_dir)
        img = place_text(img, font, "hello国", debug=debug)

        show_img(img, img_path.name)
        # break
