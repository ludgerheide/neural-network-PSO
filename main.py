from pyswarm import pso

from image_tools import *
from neural_network_tools import *

img = '/home/ludger/Downloads/faces-aligned/L/L2.png'
mask = '/home/ludger/Schreibtisch/barMask.png'

img = load_image(img, 3)
mask = load_image(mask, 1)

nn = NeuralNetworkTools()


def optimization_function(param):
    fooling_pattern = create_fooling_pattern(IMAGE_DIM, param)
    img2 = blend(img, fooling_pattern, mask)
    x = nn.calculate_likeness(img, img2)
    return x


lower_params_bound = [0, 0, 0]
upper_params_bound = [255, 255, 255]

xopt, fopt = pso(optimization_function, lower_params_bound, upper_params_bound, debug=True)

print(xopt)
print(fopt)
