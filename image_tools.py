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

    return blended_image


NUMBER_OF_LINES = 20


def create_fooling_pattern(size, param):
    """
    Generates a square fooling pattern according to the designated parameters
    :param size: the size in pixels of the fooling pattern
    :param param: an array containing the values for all the parameters
    :return: the fooling pattern (not masked yet)
    """

    fooling_pattern = np.zeros((size, size / 2, 3), dtype=np.uint8)

    # param0,1,2 is the background color
    if is_cv2():
        cv2.rectangle(fooling_pattern, (0, 0), (size, size), color=(param[0], param[1], param[2]),
                      thickness=cv2.cv.CV_FILLED)
    else:
        cv2.rectangle(fooling_pattern, (0, 0), (size, size), color=(param[0], param[1], param[2]), thickness=-1)

    # Create NUMBER_OF_LINES lines
    for i in range(0, NUMBER_OF_LINES):
        pt1 = (int(round(param[8 * i + 0 + 3])), int(round(param[8 * i + 1 + 3])))
        pt2 = (int(round(param[8 * i + 2 + 3])), int(round(param[8 * i + 3 + 3])))
        color = (param[8 * i + 4 + 3], param[8 * i + 5 + 3], param[8 * i + 6 + 3])
        if is_cv2():
            cv2.line(fooling_pattern, pt1, pt2, color, thickness=int(round(param[8 * i + 7 + 3])), lineType=cv2.CV_AA,
                     shift=0)
        else:
            cv2.line(fooling_pattern, pt1, pt2, color, thickness=int(round(param[8 * i + 7 + 3])), lineType=cv2.LINE_AA,
                     shift=0)

    # Blur it
    blur_value = int(round(param[len(param) - 1])) * 2 + 1  # Must be an odd integer
    fooling_pattern = cv2.blur(fooling_pattern, (blur_value, blur_value))

    # Now mirror it to the other side of the reulsitng image
    flipped_pattern = cv2.flip(fooling_pattern, 1)
    result_image = np.zeros((size, size, 3), dtype=np.uint8)
    result_image[0:size, 0:(size / 2), ...] = fooling_pattern
    result_image[0:size, (size / 2):size, ...] = flipped_pattern

    return result_image


def create_fooling_pattern_bounds():
    """
    Returns a 2-Array tuple containing the upper and lower bounds for create_fooling_pattern
    :return: a 2-Array tuple containing the upper and lower bounds for create_fooling_pattern
    """

    lower_params_bound = np.zeros(1 + 3 + 8 * NUMBER_OF_LINES, dtype=float)
    upper_params_bound = np.zeros(1 + 3 + 8 * NUMBER_OF_LINES, dtype=float)
    upper_params_bound[0] = 255
    upper_params_bound[1] = 255
    upper_params_bound[2] = 255
    for i in range(0, NUMBER_OF_LINES):
        upper_params_bound[8 * i + 0 + 3] = 48  # pt1x
        upper_params_bound[8 * i + 1 + 3] = 48  # pt1y
        upper_params_bound[8 * i + 2 + 3] = 48  # pt2x
        upper_params_bound[8 * i + 3 + 3] = 48  # pt2y
        upper_params_bound[8 * i + 4 + 3] = 255  # g
        upper_params_bound[8 * i + 5 + 3] = 255  # b
        upper_params_bound[8 * i + 6 + 3] = 255  # r
        upper_params_bound[8 * i + 7 + 3] = 4  # width (int)

    # Append blur mins/maxes
    lower_params_bound[len(lower_params_bound) - 1] = 0  # blur min
    upper_params_bound[len(upper_params_bound) - 1] = 10  # blur max

    return lower_params_bound, upper_params_bound


def is_cv2():
    # if we are using OpenCV 2, then our cv2.__version__ will start
    # with '2.'
    return check_opencv_version("2.")


def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")


def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib

    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major)
