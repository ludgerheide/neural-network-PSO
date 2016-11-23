import numpy as np
import time

from pyswarm import pso

import image_tools
import neural_network_tools

start = time.time()

# source_paths = {'images/source1.png', 'images/source2.png', 'images/source3.png', 'images/source4.png', 'images/source5.png'}
source_paths = {'images/source2.png', 'images/source4.png', 'images/source5.png'}
# target_paths = {'images/target1.png', 'images/target2.png', 'images/target3.png', 'images/target4.png', 'images/target5.png'}
target_paths = {'images/target1.png', 'images/target2.png', 'images/target5.png'}
mask_path = 'images/barMask.png'

source_images = list()
for source_path in source_paths:
    source_images.append(image_tools.load_image(source_path, 3))

target_images = list()
for target_path in target_paths:
    target_images.append(image_tools.load_image(target_path, 3))

mask = image_tools.load_image(mask_path, 1)

# Instantiate a reusable neural network object to increase performance
nn = neural_network_tools.NeuralNetworkTools()

##### DEFINE ALTERNATE FOOLING PATTERN GENERATORS HERE #####
fooling_generator = image_tools.create_fooling_pattern
(lower_bounds, upper_bounds) = image_tools.create_fooling_pattern_bounds()
##### END FOOLING PATTERN GENERATOR DEFINITION #####

optimization_counter = 0
optimization_start = time.time()

number_of_iterations = 2
number_of_particles = 10


def optimization_function(generator_parameters):
    """
    Optimization function. Returns the average likenss of the masked source to the target
    :param generator_parameters:
    :return: Avergae likeness (Standard deviation is ignored)
    """
    fooling_pattern = fooling_generator(neural_network_tools.IMAGE_DIM, generator_parameters)

    results = nn.calculate_likenesses(source_images, target_images, mask, fooling_pattern)

    global optimization_counter
    optimization_counter = optimization_counter + 1
    run_time = time.time() - optimization_start
    time_remaining = (run_time / optimization_counter) * (number_of_particles * (number_of_iterations + 1)) - run_time
    print('Optimization run {:5d} of {}, optimizing for {:8.1f} seconds, estimated time remaing {:8.1f} seconds'.format(
        optimization_counter, (number_of_iterations + 1) * number_of_particles, run_time, time_remaining))
    return np.mean(results)


print("Startup took {:.1f} seconds.".format(time.time() - start))
xopt, fopt = pso(optimization_function, lower_bounds, upper_bounds, debug=False, maxiter=number_of_iterations,
                 swarmsize=number_of_particles, minfunc=1e-3, phig=2.0, phip=2.0 )

print("Best:" + str(fopt))

empty_mask = np.multiply(255,
                         np.ones((neural_network_tools.IMAGE_DIM, neural_network_tools.IMAGE_DIM, 1), dtype=np.uint8))
empty_pattern = np.multiply(255, np.ones((neural_network_tools.IMAGE_DIM, neural_network_tools.IMAGE_DIM, 3),
                                         dtype=np.uint8))
likenesses = nn.calculate_likenesses(source_images, target_images, empty_mask, empty_pattern)
print("Likenesses between target and source without fooling pattern: " + str(likenesses))
print("Mean: " + str(np.mean(likenesses)) + ", Standard deviation: " + str(np.std(likenesses)))

fooling_pattern = fooling_generator(neural_network_tools.IMAGE_DIM, xopt)
image_tools.save_image(fooling_pattern, 'images/fooling_pattern' + str(fopt) + '.png')

for i in range(0, len(source_images)):
    source_img = source_images.pop()
    blended_image = image_tools.blend(source_img, fooling_pattern, mask)
    image_tools.save_image(blended_image, 'images/result' + str(fopt) + '_' + str(i) + '.png')
