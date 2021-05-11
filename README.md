# Human in loop grab cut based foreground extraction

Extracting foreground regions in image plane is very basic and primary task for many complex computer vision tasks. Extracting foreground parts from images includes extracting human part or any desired objects from the images. Grab Cut algorithm is one of the most popular method in the field of image segmentation. Grab Cut is an iterative method. Further, the addition of manual seed points can boost the performance of the grab cut algorithm. While Grab cut can directly be used using OpenCV APIs but manually adding seed points for GrabCut is not straight forward. Hence, this repo aims at developing a pipeline to manually add the seed points (which includes marking of background region and foreground regions) in the image to extract the foregroiund regions. 

**Dependensies**
* Python 3.6+
* OpenCV 4.5


**Steps**
* Clone this repository
* `cd \path\to\the\repository`
* Put your test images in the folder `/images`
* Type in the terminal `python3 main.py`
* Segmented part will be saved in the folder `/foreground_mask`
* A (video link)(https://youtu.be/UtwnMofyMqs) is attached which demostrated how to feed the seed points (foreground and background pixels).


<p align="center">
  <img src="results/resut_img.jpg" />
</p>


**Observations**
* Has been tested for simple cases, where the background is uniform and has a contract difference with target object.
