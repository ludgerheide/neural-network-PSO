import argparse
import numpy as np
import time

from pyswarm import pso

import image_tools
import neural_network_tools


def optimization_function_dodging(generator_parameters):
    """
    Optimization function. Returns the average likenss of the masked source to the source
    :param generator_parameters:
    :return: -1 * Avergae likeness
    """
    local_fooling_pattern = fooling_generator(neural_network_tools.IMAGE_DIM, generator_parameters)

    results = nn.calculate_likenesses(source_images, source_images, mask, local_fooling_pattern)

    global optimization_counter
    optimization_counter += 1
    run_time = time.time() - optimization_start
    time_remaining = (run_time / optimization_counter) * (number_of_particles * (number_of_iterations + 1)) - run_time
    print('Optimization run {:5d} of {}, optimizing for {:8.1f} seconds, estimated time remaing {:8.1f} seconds'.format(
        optimization_counter, (number_of_iterations + 1) * number_of_particles, run_time, time_remaining))
    return -1 * np.mean(results)


def optimization_function_fooling(generator_parameters):
    """
    Optimization function. Returns the average likenss of the masked source to the target
    :param generator_parameters:
    :return: Avergae likeness (Standard deviation is ignored)
    """
    local_fooling_pattern = fooling_generator(neural_network_tools.IMAGE_DIM, generator_parameters)

    results = nn.calculate_likenesses(source_images, target_images, mask, local_fooling_pattern)

    global optimization_counter
    optimization_counter += 1
    run_time = time.time() - optimization_start
    time_remaining = (run_time / optimization_counter) * (number_of_particles * (number_of_iterations + 1)) - run_time
    print('Optimization run {:5d} of {}, optimizing for {:8.1f} seconds, estimated time remaing {:8.1f} seconds'.format(
        optimization_counter, (number_of_iterations + 1) * number_of_particles, run_time, time_remaining))
    return np.mean(results)


# Main program start: Argument paring
start = time.time()

# Read in arguments and paths and define optikmization settings
source_paths = ['images/source2.png', 'images/source4.png', 'images/source5.png', 'images/source1.png',
                'images/source3.png']
target_paths = ['images/target1.png', 'images/target2.png', 'images/target5.png', 'images/target3.png',
                'images/target4.png']
mask_path = 'images/barMask.png'

source_images = list()
target_images = list()
for i in range(0, 5):
    source_images.append(image_tools.load_image(source_paths[i], 3))
    target_images.append(image_tools.load_image(target_paths[i], 3))

mask = image_tools.load_image(mask_path, 1)

# Instantiate a reusable neural network object to increase performance
nn = neural_network_tools.NeuralNetworkTools()

##### DEFINE ALTERNATE FOOLING PATTERN GENERATORS HERE #####
fooling_generator = image_tools.create_fooling_pattern
(lower_bounds, upper_bounds) = image_tools.create_fooling_pattern_bounds()
##### END FOOLING PATTERN GENERATOR DEFINITION #####

print("rep of face 5")
x = nn.net.forward(source_images[2])
print(str(x))

empty_mask = np.multiply(0,
                         np.ones((neural_network_tools.IMAGE_DIM, neural_network_tools.IMAGE_DIM, 1), dtype=np.uint8))
empty_pattern = np.multiply(255, np.ones((neural_network_tools.IMAGE_DIM, neural_network_tools.IMAGE_DIM, 3),
                                         dtype=np.uint8))

likenesses_target_target = nn.calculate_likenesses(target_images, target_images, empty_mask, empty_pattern)
print("Likenesses between target and itself:\n" + str(likenesses_target_target))
print("Mean: " + str(np.mean(likenesses_target_target)) + ", Standard deviation: " + str(
    np.std(likenesses_target_target)) + "\n")

likenesses_source_source = nn.calculate_likenesses(source_images, source_images, empty_mask, empty_pattern)
print("Likenesses between source and itself:\n" + str(likenesses_source_source))
print("Mean: " + str(np.mean(likenesses_source_source)) + ", Standard deviation: " + str(
    np.std(likenesses_source_source)) + "\n")

likenesses_source_target = nn.calculate_likenesses(source_images, target_images, empty_mask, empty_pattern)
print("Likenesses between target and source:\n" + str(likenesses_source_target))
print("Mean: " + str(np.mean(likenesses_source_target)) + ", Standard deviation: " + str(
    np.std(likenesses_source_target)) + "\n")

