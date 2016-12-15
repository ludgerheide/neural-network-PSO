Implementation (Core code and annotations)
==========================================

In order to implement the impersonation attack, we utilize the
python[^1] programming language, the openFace[^2] face recognition API
that implements the FaceNet[^3] face recognition neural network.
Additionally, the pyswarm[^4] and openCV[^5] packages are used for
particle swarm optimization and pattern generation, respectively.

In order to run the code, the required packages were manually installed
on a virtual machine running ubuntu Linux. To replicate our results, the
reader is advised to contact us and we will provide the prepackaged
virtual machine image. It is also possible to install the prerequisites
on a different computer. However due to the large downloads required and
the poor Internet connectivity in China, this is not advised.

Our code is available on github[^6], it should also be included as a
.zip file with this report. The following section gives a high-level
overview over the functionality. For detailed documentation on the
different functions, the reader is referred to the source code and the
attached pydoc documentation in the next section.

Configuration
-------------

In addition to having the prerequisites installed, running the software
requires the user to define the abolute paths to the face recognition
and face detection neural networks. This is done by copying
config-sample.json to config.json and editing the contents in order to
point to the correct files (they should be in the model directory of
your openFace installation).

Program argumenhts
------------------

The code is run by navigating to the project directory and invoking
main.py, with the following arguments allowed:

-   --particlesThe number of particles used by the PSO algorithm.
    Improving the number improves result quality, however this comes
    with a linear increase in computing time.\
    default: 100
-   --iterationsThe number of iterations the PSO algorithm runs.
    Improving the number improves result quality, however this comes
    with a linear increase in computing time.\
    default: 10 (this gives acceptable runtime, for acceptable results a
    value of 100 is recommended)
-   --imagesThe number of images to compare with each other. For a value
    of 1, one source image is compared to one target image, for larger
    numbers, each source image is comparer with each target image. As
    such, an increase in this number comes with a *quadratic* increase
    in computing time.\
    default: 1
-   --dodgingEnables dodging mode. Instead of making the source look
    like the target, it tries to maximize the distance between the
    source and target images. This mode is as if yet untested.

Results
-------

A run is ended when the maximum number of iteratitons has been reached
or the change between iterations has become insignificant. After the
end, three textual results and two types of result images are created.

The first textual output is the likeness between the target and itself.
For a single-image run, this is always 0.0, however for a multi-image
run this provides a baseline showing how similar images of the same face
appear to the neural network.

The next output shows the likeness between target and source without the
fooling pattern being used. This should be a number around 2.0.

The final output is the likeness between target and source using the
fooling pattern overlaid on the source image. In a perfect result, this
should be identical to to the likeness between the target and itself
(the first output). In practice, values between 0.6-0.7 can be achieved
with enough particles and iterations.

Furthermore, the fooling pattern as well as each source image overlaid
with the fooling pattern are saved to the folder images/. The file names
are

-   fooling\_*resultValue*\_*imageNumber*\_result.png
-   fooling\_*resultValue*\_fooling\_pattern.png.

Further considerations
----------------------

As the pyswarm package uses true random start values for the particles,
each run will result in slightly different results.

The images used were pre-aligned using the dlib face predictor, using
the method shown in section 2 of the openFace documentation[^7].

Currently, the program expects all source and target images to be of the
size of 96 x 96 x 3 (color channels) pixels. This can be changed by
changing the value of IMAGE\_DIM in neural\_network\_tools.py. This
should only be done if using a different neural network that expects a
different image size.

The code has been prepared to use different fooling pattern generators,
however this is not yet implemented as a fully modular solution. A
fooling pattern generator should have a main method that generates a
pattern of the size IMAGE\_DIM x IMAGE\_DIM x 3 (color channels) using
an argument array. Furthermore, it should have a helper function that
returns a tuple of two arrays that represent the minimum and maximum
values for each parameter of the parameter array. This generator can
then be exchanged for the default one in the lines 82 and 83 of main.py.
It is recommended to create a conditional statement and command line
switch for this.

The shape of the fooling pattern overlay can be modified by editing the
file images/barMask.png with an image editor. Different gray levels for
advanced blending are supported, however the file must remain grayscale
without an alpha channel.

Pydoc documentation
===================

### image\_tools.py

[]{#anchor}**blend**(background, foreground, mask)

*Blends a foregorund image onto a background image according to a mask.\
Where the mask value is 255, only the foreground image is used in the resulting image\
Where the mask value is 0, only the background image is used.\
With a value in between, the result is (maskValue/255) \* newImage\
                                     + ((255-maskValue) / 255) oldImage\
:param background: sciPy array containing the background. can be 1d or 3d\
:param foreground: sciPy array containing the foreground. can be 1d or 3d\
:param mask: sciPy arraz containing the mask. cmust be 1d\
:return: the overlaid image*

[]{#anchor-1}**check\_opencv\_version**(major, lib=None)

*helper function to find openCV version*

[]{#anchor-2}**create\_fooling\_pattern**(size, param)

*Generates a square fooling pattern according to the designated parameters\
:param size: the size in pixels of the fooling pattern\
:param param: an array containing the values for all the parameters\
:return: the fooling pattern (not masked yet)*

[]{#anchor-3}**create\_fooling\_pattern\_bounds**()

*Returns a 2-Array tuple containing the upper and lower bounds for create\_fooling\_pattern\
:return: a 2-Array tuple containing the upper and lower bounds for create\_fooling\_pattern*

[]{#anchor-4}**is\_cv2**()

*helper function to find if openCV version is 2*

[]{#anchor-5}**is\_cv3**()

*helper function to find if openCV version is 3*

[]{#anchor-6}**load\_image**(filename, layers)

*Loads an image into a sciPy array. Assures that it's size matches our size constant\
:param filename: the complete path to the file\
:param layers: the number of layers. Allowed: 1 (greyscale), 3 (RGB)\
:return: a sciPy array containing the image*

[]{#anchor-7}**save\_image**(image, filename)

*Saves an Image to a png file\
:param filename: the complete path and filename\
:param image: The Image array. Can be one- or three-dimensional\
:return: nothing*

### neural\_network\_tools.py

*\# Code from demos/compare.py by the openFace project\
\#\
\# Copyright 2015-2016 Carnegie Mellon University\
\# Modifications 2016 by Ludger Heide\
\#\
\# Licensed under the Apache License, Version 2.0 (the "License");\
\# you may not use this file except in compliance with the License.\
\# You may obtain a copy of the License at\
\#\
\#     *[*http://www.apache.org/licenses/LICENSE-2.0*](http://www.apache.org/licenses/LICENSE-2.0)*\
\#\
\# Unless required by applicable law or agreed to in writing, software\
\# distributed under the License is distributed on an "AS IS" BASIS,\
\# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\
\# See the License for the specific language governing permissions and\
\# limitations under the License.*

[]{#anchor-8}***\_\_init\_\_**(self)*

[]{#anchor-9}**align\_face**(self, img)

*Aligns a face found in an image and crops it to 96x96\
:param img:\
:return: The aligned image*

[]{#anchor-10}**calculate\_likeness**(self, img1, img2)

*Calculates how close the images are to each other for the face recognition DNN\
:param img1: first image\
:param img2: second image\
:return: a number between 0 (same image) and a high value (\~2) for different faces*

[]{#anchor-11}**calculate\_likenesses**(self, sources, targets, mask,
fooling\_pattern)

*Calculates how close the images are to each other for the face recognition DNN\
:param fooling\_pattern: The fooling pattern that is overliad on the source images\
:param sources: Source images (they get masked)\
:param targets: Target images\
:param mask: The mask\
:return: a number between 0 (same image) and a high value (\~2) for different faces*

### **tools.py**

[]{#anchor-12}******load\_key\_from\_config******(key)**

*Loads a key from config.json\
:parameter key the key to load (a string)\
:return:*

[^1]: https://www.python.org/downloads/release/python-2712/

[^2]: B. Amos, B. Ludwiczuk, M. Satyanarayanan, *„Openface: A
    general-purpose face recognition library with mobile applications“*,
    CMU-CS-16-118, CMU School of Computer Science, Tech. Rep., 2016.

[^3]: Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
    „*Facenet: A unified embedding for face recognition and
    clustering.“* Proceedings of the IEEE Conference on Computer Vision
    and Pattern Recognition. 2015.

[^4]: https://pythonhosted.org/pyswarm/

[^5]: http://opencv.org/

[^6]: https://github.com/ludgerheide/neural-network-PSO

[^7]: https://cmusatyalab.github.io/openface/visualizations/
