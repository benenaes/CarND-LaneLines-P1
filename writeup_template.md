---
typora-root-url: writeup
---

# **Finding Lane Lines on the Road** 

## Benedikt Naessens

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The pipeline consists of the following steps:

- Conversion to a grayscale image

![blurred](/grayscale.jpg)

- Apply Gaussian blur onto the grayscale image (kernel  = 5)

![blurred](/blurred.jpg)

- Apply adaptive Canny edge detection on the blurred image (low and high thresholds are automatically deduced using a fixed deviation from the median gray pixel value)

![canny_edge](/canny_edge.jpg)

- Calculate a trapezoid region of interest ('ROI') based on some fixed parameters and mask the Canny edge image with it

![masked_roi](/masked_roi.jpg)

- Apply a Hough transform on the masked image to get a set of line segments that are candidates for the left and right lane lines

![hough_lines](/hough_lines.jpg)

- Select which lines belong to the left and right lane lines based on the following actions:
  - prune horizontal slope lines
  - divide into potential left lane and right lane lines using the slope of the lines
  - calculate weighted mean of left lane and right lane slopes (weights according to how low in the screen the lines are)
  - if a slope of a line is too much off from the weighted mean, prune it away

  ![SelectedLeftAndRightLaneLines](/SelectedLeftAndRightLaneLines.jpg)


- For each lane line: 
  - Calculate the slope and intercept of the line that fits the line segments that were binned into the group of the lane line (first degree polynomial least square fitting)
  - Draw a red line with given slope and intercept in the trapezoid ROI (using the top and bottom Y coordinates of the ROI)
- Blend the original image with the image with red lane lines

![Blended](/Blended.jpg)

Some other noteworthy functions/classes are:

- *draw_lines*(): just draws an array of lines on an image (I have chosen not to extend this function and put all calculations in other functions)
- *InvertedYCoordinateSystem* class: An alternative image coordinate system where the bottom left corner of the image is the origin. This coordinate system can be more natural to deal with mathematical or geometric terms like e.g. slopes. It is not necessary at all to have this class to complete the project, but by adding it, I found the code easier to read.

The code could also be run separately on the test images with *lane_test.py*

### 2. Identify potential shortcomings with your current pipeline

Shortcomings of the project:

- The ROI is trapezoid: it could be even more useful to have two separate polygons for each lane line to get rid of any potential noise in the middle
- The ROI needs to be defined manually: it could be estimated from intrinsic and extrinsic parameters (field of view, focus, zoom, orientation, position, ...). Also, the ROI could be adapted during the ride based on feedback from the lane detection
- The adaptive Canny edge detection is still naïve: it copes very badly with changing light or different concrete colours.
- Some parameters of the lane line segment selection are manually set (minimum absolute slopes)
- The "challenge" shows a lot of difficulties when the lane lines are not straight: first degree polynomial lane fitting doesn't suffice here


### 3. Suggest possible improvements to your pipeline

Possible improvements:

- Use HSV conversion instead of grayscale conversion to find the yellow and white lane lines. On the other hand, this has the disadvantage that the algorithm could not be used in the absence of lane lines (e.g. when there is grass, rocks or sand next to the road without any lane lines). Alternatively, HSV conversion could also be used to "detect" road pixel candidates (colours resembling concrete) and in that way define a ROI automatically.
- Improved Canny edge detection algorithms, for example: http://cvrs.whu.edu.cn/projects/cannyLines/
- Use the previously detected lane lines to smoothen the result of the current pipeline. The tricky part is finding good smoothening factors to make sure the lane detection doesn't lag too much in case of sharper turns.
- Higher degree polynomial lane line fitting