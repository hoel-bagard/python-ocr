"""Poisson seamless cloning.

TODO: move this to the dataset readme.md.
I would recommend first reading the paper: "Poisson Image Editing" by Patrick Perez, Michel Gangnet, Andrew Blake.
(can be found here: https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
Then this blog post to help clarify things: https://erkaman.github.io/posts/poisson_blending.html,
and then go back to the original paper.

Code is taken and slightly modified from these two repos:
https://github.com/gachiemchiep/SynthText/blob/master/poisson_reconstruct.py
"""
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve

from src.utils.misc import show_img


def poisson_edit_old(source: np.ndarray, target: np.ndarray, mask: np.ndarray, offset: tuple[int, int]):
    """The poisson blending function.

    Code from https://github.com/PPPW/poisson-image-editing/blob/master/poisson_image_editing.py
    The target's size needs to be >= to the source's.
    The target's size and the mask's need to be the same.

    See the Discrete Poisson equation wikipedia article for the notation.

    Args:
        source: The source image (image with the object to insert).
        target: The target image (image that serves as background).
        mask: The mask of the object to insert.
        offset: ???

    Return:
        The target image with the source image blended into it according to the mask.
    """
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min

    mat_m = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    source = cv2.warpAffine(source, mat_m, (x_range, y_range))

    def laplacian_matrix(n: int, m: int):
        """Generate the Poisson matrix.

        Refer to:
        https://en.wikipedia.org/wiki/Discrete_Poisson_equation
        Note: it's the transpose of the wiki's matrix

        Args:
            n (int):
            m (int):

        Returns:
            The matrix TODO
        """
        mat_d = scipy.sparse.lil_matrix((m, m))
        mat_d.setdiag(-1, -1)
        mat_d.setdiag(4)
        mat_d.setdiag(-1, 1)

        mat_a = scipy.sparse.block_diag([mat_d] * n).tolil()

        mat_a.setdiag(-1, 1*m)
        mat_a.setdiag(-1, -1*m)

        return mat_a

    mat_a = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_a.tocsc()

    # set the region outside the mask to identity
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_a[k, k] = 1
                mat_a[k, k + 1] = 0
                mat_a[k, k - 1] = 0
                mat_a[k, k + x_range] = 0
                mat_a[k, k - x_range] = 0

    mat_a = mat_a.tocsc()

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 0.8
        mat_b = laplacian.dot(source_flat)*alpha + laplacian.dot(target_flat)*(1-alpha)

        # outside the mask:
        # f = t
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(mat_a, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype("uint8")

        target[y_min:y_max, x_min:x_max, channel] = x

    return target


def get_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the the x and y partial derivatives of the given grayscale image.

    Args:
        img (np.ndarray): The grayscale image whose gradient is to be computed.

    Returns:
        A tuple, the partial derivatives along the x and y axis.
    """
    height, width = img.shape
    grad_x, grad_y = np.zeros((height, width), dtype="float32"), np.zeros((height, width), dtype="float32")
    i, j = np.expand_dims(np.arange(0, height-1), -1), np.arange(0, width-1)
    grad_x[i, j] = img[i, j+1] - img[i, j]
    grad_y[i, j] = img[i+1, j] - img[i, j]
    return grad_x, grad_y


def get_laplacian(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """Compute the the x and y laplacian, given the x and y gradients.

    Args:
        grad_x (np.ndarray): The partial derivative along the x axis.
        grad_y (np.ndarray): The partial derivative along the y axis.

    Returns:
        The laplacian corresponding to the given derivatives.
    """
    height, width = grad_x.shape
    laplacian_x, laplacian_y = np.zeros((height, width), dtype="float32"), np.zeros((height, width), dtype="float32")
    i, j = np.expand_dims(np.arange(0, height-1), -1), np.arange(0, width-1)
    laplacian_x[i, j+1] = grad_x[i, j+1] - grad_x[i, j]
    laplacian_y[i+1, j] = grad_y[i+1, j] - grad_y[i, j]
    return laplacian_x + laplacian_y


def poisson_edit(source_img: np.ndarray, target_img: np.ndarray, scale_grad=1.0, mode='max'):
    """Combine the two input images using poission editing.

    The images should be of the same size.

    Args:
        source_img (np.ndarray): The source image (image with the object to insert).
        target_img (np.ndarray): The target image (image that serves as background).

    Returns:
        The resulting image.
    """
    assert np.all(source_img.shape == target_img.shape)

    source_img = source_img.copy().astype("float32")
    target_img = target_img.copy().astype("float32")
    result_img = np.zeros_like(source_img)

    # frac of gradients which come from source:
    for channel in range(source_img.shape[2]):
        source_grad_x, source_grad_y = get_gradients(source_img[:, :, channel])
        target_grad_x, target_grad_y = get_gradients(target_img[:, :, channel])

        # Optionally scale the gradients
        source_grad_x *= scale_grad
        source_grad_y *= scale_grad

        gxs_idx = source_grad_x != 0
        source_grad_y_idx = source_grad_y != 0

        # mix the source and target gradients:
        if mode == 'max':
            # Find all the spots where the target (background) gradient is bigger than the source one.
            # This allows keeping the relief of the background image.
            grad_x = source_grad_x.copy()  # TODO: copy needed ?
            grad_x_max = (np.abs(target_grad_x)) > np.abs(source_grad_x)
            grad_x[grad_x_max] = target_grad_x[grad_x_max]

            grad_y = source_grad_y.copy()
            gym = np.abs(target_grad_y) > np.abs(source_grad_y)
            grad_y[gym] = target_grad_y[gym]

            # get gradient mixture statistics:
            f_grad_x = np.sum((grad_x[gxs_idx] == source_grad_x[gxs_idx]).flat) / (np.sum(gxs_idx.flat)+1e-6)
            f_grad_y = np.sum((grad_y[source_grad_y_idx] == source_grad_y[source_grad_y_idx]).flat) / (np.sum(source_grad_y_idx.flat)+1e-6)
            if min(f_grad_x, f_grad_y) <= 0.35:
                m = 'max'
                if scale_grad > 1:
                    m = 'blend'
                return poisson_edit(source_img, target_img, scale_grad=1.5, mode=m)

        elif mode == 'blend':  # from recursive call:
            # just do an alpha blend
            grad_x = source_grad_x+target_grad_x
            grad_y = source_grad_y+target_grad_y

    #     result_img[:, :, channel] = np.clip(poisson_solve(gx, gy, imd), 0, 255)

    # return result_img.astype('uint8')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=("Script to merge to images using poissong editing."
                                         "Run with 'python -m src.dataset.poisson_merging <path>'."))
    parser.add_argument("target_img_path", type=Path, help="Path to the target image (i.e. the background).")
    parser.add_argument("--source_img_path", "-s", type=Path, default=Path("data/poisson_source.jpg"),
                        help="Path to the source image (part to copy/paste).")
    parser.add_argument("--mask_img_path", "-m", type=Path, default=Path("data/poisson_mask.png"),
                        help="Path to the mask corresponding to the source image.")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    args = parser.parse_args()

    target_img_path: Path = args.target_img_path
    source_img_path: Path = args.source_img_path
    mask_img_path: Path = args.mask_img_path

    source_img = cv2.imread(str(source_img_path))
    target_img = cv2.imread(str(target_img_path))
    mask = cv2.imread(str(mask_img_path), cv2.IMREAD_GRAYSCALE)

    # Temp
    target_img = cv2.resize(target_img, source_img.shape[:2][::-1], interpolation=cv2.INTER_AREA)

    poisson_edit(source_img, target_img)

    # The mask is expected to have the same shape as the target.
    # # mask = cv2.resize(mask, target_img.shape[:2][::-1], interpolation=cv2.INTER_AREA)
    # mask[mask != 0] = 1  # Make mask binary
    # offset = (0, 66)
    # result = poisson_edit(source_img, target_img, mask, offset)

    # show_img(result)
