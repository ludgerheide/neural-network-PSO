from pyswarm import pso

from image_tools import *
from neural_network_tools import *

source_path = 'images/source.png'
target_path = 'images/target.png'
mask_path = 'images/barMask.png'

source = load_image(source_path, 3)
target = load_image(target_path, 3)
mask = load_image(mask_path, 1)

nn = NeuralNetworkTools()


def generate_image(param):
    """
    Generates the image that is compared to the other image in optimization_function
    :param param: The parameter array for generate_fooling_pattern
    :return: an image object
    """
    fooling_pattern = create_fooling_pattern(IMAGE_DIM, param)
    blended_image = blend(source, fooling_pattern, mask)
    return blended_image


def optimization_function(param):
    fooling_image = generate_image(param)
    x = nn.calculate_likeness(target, fooling_image)
    return x


def constraint_function(param):
    """
    Constraint function makes sure whatever we generate actually looks like a face
    :param param: the parameters to generate the fooling pattern
    :return: -1 if it is not a face, 1 if it is a face
    """
    img = generate_image(param)
    bb = nn.align.getLargestFaceBoundingBox(img)
    if bb is None:
        return [-1]
    else:
        return [1]


lower_params_bound = np.zeros(3 + 8 * NUMBER_OF_LINES, dtype=float)

upper_params_bound = np.zeros(3 + 8 * NUMBER_OF_LINES, dtype=float)
upper_params_bound[0] = 255
upper_params_bound[1] = 255
upper_params_bound[2] = 255
for i in range(0, NUMBER_OF_LINES):
    upper_params_bound[8 * i + 0 + 3] = 96  # pt1x
    upper_params_bound[8 * i + 1 + 3] = 96  # pt1y
    upper_params_bound[8 * i + 2 + 3] = 96  # pt2x
    upper_params_bound[8 * i + 3 + 3] = 96  # pt2y
    upper_params_bound[8 * i + 4 + 3] = 255  # g
    upper_params_bound[8 * i + 5 + 3] = 255  # b
    upper_params_bound[8 * i + 6 + 3] = 255  # r
    upper_params_bound[8 * i + 7 + 3] = 4  # width (int)

xopt, fopt = pso(optimization_function, lower_params_bound, upper_params_bound, f_ieqcons=constraint_function, phip=1.0,
                 phig=1.0, debug=False, maxiter=100, swarmsize=100, minfunc=1e-3)

print("Best:" + str(fopt))
print("Likeness between target and source without fooling pattern: " + str(nn.calculate_likeness(source, target)))

pattern = create_fooling_pattern(IMAGE_DIM, xopt)
save_image(pattern, 'images/fooling_pattern' + str(fopt) + '.png')

res = generate_image(xopt)
save_image(res, 'images/result' + str(fopt) + '.png')
