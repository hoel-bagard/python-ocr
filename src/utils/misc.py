from shutil import get_terminal_size
from typing import Optional

import cv2
import numpy as np


def clean_print(msg: str, fallback: Optional[tuple[int, int]] = (156, 38), end='\n'):
    r"""Function that prints the given string to the console and erases any previous print made on the same line.

    Args:
        msg (str): String to print to the console
        fallback (tuple, optional): Size of the terminal to use if it cannot be determined by shutil
                                    (if using windows for example)
        end (str): What to add at the end of the print. Usually '\n' (new line), or '\r' (back to the start of the line)
    """
    print(msg + ' ' * (get_terminal_size(fallback=fallback).columns - len(msg)), end=end, flush=True)


def show_img(img: np.ndarray, window_name: str = "Image"):
    """Displays an image until the user presses the "q" key.

    Args:
        img: The image that is to be displayed.
        window_name (str): The name of the window in which the image will be displayed.
    """
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
