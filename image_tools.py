import numpy as np
from scipy import misc

import cv2


def load_image(filename, layers):
    """
    Loads an image into a sciPy array. Assures that it's size matches our size constant
    :param filename: the complete path to the file
    :param layers: the number of layers. Allowed: 1 (greyscale), 3 (RGB)
    :return: a sciPy array containing the image
    """
    if layers == 3:
        img = misc.imread(filename, mode='RGB')
    elif layers == 1:
        img = misc.imread(filename, flatten=True, mode='L')
        img.shape = (img.shape[0], img.shape[1], 1)
    else:
        raise NotImplementedError('Invalid layer count!')

    return img


def save_image(image, filename):
    """
    Saves an Image to a png file
    :param filename: the complete path and filename
    :param image: The Image array. Can be one- or three-dimensional
    :return: nothing
    """
    misc.imsave(filename, image)


def blend(background, foreground, mask):
    """
    Blends a foregorund image onto a background image according to a mask.
    Where the mask value is 255, only the foreground image is used in the resulting image
    Where the mask value is 0, only the background image is used.
    With a value in between, the result is (maskValue/255) * newImage
                                         + ((255-maskValue) / 255) oldImage
    :param background: sciPy array containing the background. can be 1d or 3d
    :param foreground: sciPy array containing the foreground. can be 1d or 3d
    :param mask: sciPy arraz containing the mask. cmust be 1d
    :return: the overlaid image
    """

    # The images must be if equal size
    if background.shape != foreground.shape:
        raise ValueError('Foregrouns and background are not the same size or dimension!')
    if background.shape[0] != mask.shape[0] or background.shape[1] != mask.shape[1]:
        raise ValueError('Foreground/background and mask are not the same size!')

    # Create the array for the new img
    blended_image = np.zeros(background.shape, dtype=np.uint8)
    mask.shape = background.shape[0], background.shape[1]

    for dimension in range(0, background.shape[2]):
        blended_image[..., dimension] = \
            np.multiply(np.divide(mask, 255.0), foreground[..., dimension]) \
            + np.multiply(np.divide((255 - mask), 255.0), background[..., dimension])

        # # Save the dimensions and reshape to arrays
        # old_shape = background.shape
        # background.shape = (old_shape[0] * old_shape[1], 1, old_shape[2])
        # foreground.shape = (old_shape[0] * old_shape[1], 1, old_shape[2])
        # mask.shape = (old_shape[0] * old_shape[1], 1, 1)
        #
        # #Create the array for the new img
        # blended_image = np.zeros(background.shape, dtype=np.uint8)
        #
        # # Iterate over the array
        # for dimension in range(0, old_shape[2]):
        #     for pixel in range(0, old_shape[0] * old_shape[1]):
        #         blended_image[pixel, 0, dimension] = \
        #             (mask[pixel, 0, 0] / 255.0) * foreground[pixel, 0, dimension]
        #         + (255 - mask[pixel, 0, 0] / 255.0) * background[pixel, 0, dimension]
        #
        # blended_image.shape = old_shape
    return blended_image


def create_fooling_pattern(size, param):
    """
    Generates a square fooling pattern according to the designated parameters
    :param size: the size in pixels of the fooling pattern
    :param param: an array containing the values for all the parameters
    :return: the fooling pattern (not masked yet)
    """

    fooling_pattern = np.zeros((size, size, 3), dtype=np.uint8)

    # param0,1,2 is the background color
    cv2.rectangle(fooling_pattern, (0, 0), (size, size), color=(param[0], param[1], param[2]),
                  thickness=cv2.cv.CV_FILLED)

    return fooling_pattern
