#
# Code from demos/compare.py by the openFace project
#
# Copyright 2015-2016 Carnegie Mellon University
# Modifications 2016 by Ludger Heide
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import openface

import image_tools
import tools

IMAGE_DIM = 96
DLIB_PATH_KEY = 'dlibFacePredictorPath'
MODEL_PATH_KEY = 'networkModelPath'


class NeuralNetworkTools:
    def __init__(self):
        # Load the path to dlib from config
        dlib_face_predictor_directory = tools.load_key_from_config(DLIB_PATH_KEY).encode('ascii', 'ignore')
        self.align = openface.AlignDlib(dlib_face_predictor_directory)

        # Load the path to the model from config and load the model
        network_model_directory = tools.load_key_from_config(MODEL_PATH_KEY)
        self.net = openface.TorchNeuralNet(network_model_directory, IMAGE_DIM)

    def calculate_likeness(self, img1, img2):
        """
        Calculates how close the images are to each other for the face recognition DNN
        :param img1: first image
        :param img2: second image
        :return: a number between 0 (same image) and a high value (~2) for different faces
        """

        assert img1.shape[0] == IMAGE_DIM and img1.shape[1] == IMAGE_DIM
        assert img2.shape[0] == IMAGE_DIM and img2.shape[1] == IMAGE_DIM

        rep1 = self.net.forward(img1)
        rep2 = self.net.forward(img2)

        d = rep1 - rep2
        return np.dot(d, d)

    def calculate_likenesses(self, sources, targets, mask, fooling_pattern):
        """
        Calculates how close the images are to each other for the face recognition DNN
        :param sources: Source images (they get masked)
        :param targets: Target images
        :param mask: The mask
        :return: a number between 0 (same image) and a high value (~2) for different faces
        """
        results = list()

        for target_image in targets:
            for source_image in sources:
                blended_image = image_tools.blend(source_image, fooling_pattern, mask)
                result = self.calculate_likeness(target_image, blended_image)
                results.append(result)

        return results

    def align_face(self, img):
        """
        Aligns a face found in an image and crops it to 96x96
        :param img:
        :return: The aligned image
        """

        bb = self.align.getLargestFaceBoundingBox(img)
        if bb is None:
            raise Exception("Unable to find a face!")

        aligned_face = self.align.align(IMAGE_DIM, img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align image!")

        return aligned_face
