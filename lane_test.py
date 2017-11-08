#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def adaptive_canny(image, sigma):
    """
    Canny edge detection where the low and high threshold are automatically calculated
    using the median and a deviation (sigma) from the median
    :param image: Original (grayscale) image
    :param sigma: Deviation from the median for the low and high thresholds
    :return: Image with the edges from  within the original image
    """
    median = np.median(image)

    lowThreshold = int(max(0, (1.0 - sigma) * median))
    highThreshold = int(min(255, (1.0 + sigma) * median))
    cannyEdgeImage = cv2.Canny(image, lowThreshold, highThreshold)
    return cannyEdgeImage


class LineWithSlope:
    """
    Helper class to store the slope with two coordinates that define the line
    """
    def __init__(self, x1, y1, x2, y2, slope):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.slope = slope

    def draw_line(self, img, color, thickness=2):
        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)


class InvertedYCoordinateSystem:
    """
    An alternative image coordinate system where the bottom left corner of the image is the origin
    This coordinate system can be more natural to deal with mathematical or geometric terms like e.g. slopes
    """
    def __init__(self, bottomY):
        """
        Initialization
        :param bottomY: bottom left corner of the image (X coordinates are kept the same)
        """
        self.bottomY = bottomY

    def convert_y_coord(self, y):
        """
        Convert a Y coordinate to a coordinate system
        :param y: The Y coordinate to be converted
        :return: The converted Y coordinate
        """
        return self.bottomY - y

    def calculate_slope(self, x1, y1, x2, y2):
        """
        Calculate the slope of a line in between image points (x1, y1) and (x2,y2) in the alternative coordinate system
        :param x1: X coordinate of the first point
        :param y1: (original image) Y coordinate of the first point
        :param x2: X coordinate of the second point
        :param y2: (original image) Y coordinate of the second point
        :return: The slope of the line in the alternative coordinate system
        """
        return (y1 - y2) / (x2 - x1)

    def calculate_intercept(self, slope, x1, y1):
        """
        Calculates the intercept of a line using the slope and one point (x1, y1)
        :param slope: Slope of the line in the alternative coordinate system
        :param x1: X coordinate of the point on the line
        :param y1: (original image) Y coordinate of the point on the line
        :return: The intercept of the line in the alternative coordinate system
        """
        # converted_y1 = slope * x1 + intercept
        return self.convert_y_coord(y1) - slope * x1

    def calculate_x_on_line(self, slope, intercept, y):
        """
        Calculates the x coordinate for a given Y coordinate on the line defined by given slope and intercept
        :param slope: Slope of the line in the alternative coordinate system
        :param intercept: Intercept of the line in the alternative coordinate system
        :param y: Y (image) coordinate
        :return: The X (image) coordinate
        """
        return int((self.convert_y_coord(y) - intercept) / slope)

    def select_lines(self, lines, min_abs_left_slope, min_abs_right_slope):
        """
        Select the line segments that were detected by the Hough transform and bin them in the left lane
        or the right lane group (or no group if they don't meet certain conditions)
        1. prune horizontal slope lines
        2. divide into potential left lane and right lane lines according to the line slopes
        3. calculate weighted mean of left lane and right lane slopes
          (weights according to how low in the screen the lines are)
        4. if a slope of a line is too much off from the weighted mean, prune it away
        :param lines: The line segments that were detected by the Hough transform
        :param min_abs_left_slope: Minimum absolute slope to be a candidate for the left lane segments
        :param min_abs_right_slope: Minimum absolute slope to be a candidate for the right lane segments
        :return: Two groups of lines: left lane and right lane lines
        """
        leftLaneLines, rightLaneLines = [], []
        # Steps 1 and 2
        for line in lines:
            slope = self.calculate_slope(line[0][0], line[0][1], line[0][2], line[0][3])
            if slope < 0:
                if -slope >= min_abs_right_slope:
                    rightLaneLines.append(LineWithSlope(line[0][0], line[0][1], line[0][2], line[0][3], slope))
            else:
                if slope > min_abs_left_slope:
                    leftLaneLines.append(LineWithSlope(line[0][0], line[0][1], line[0][2], line[0][3], slope))

        # Step 3: left
        totalWeight = 0
        weights = []
        avgSlope = 0
        prunedLeftLaneLines = []
        for lineWithSlope in leftLaneLines:
            bottomLine = max(lineWithSlope.y1, lineWithSlope.y2)
            weight = bottomLine / self.bottomY
            totalWeight += weight
            weights.append(weight)
        for idx, lineWithSlope in enumerate(leftLaneLines):
            avgSlope += (lineWithSlope.slope * (weights[idx] / totalWeight))
        # Step 4: left
        for lineWithSlope in leftLaneLines:
            if (abs(lineWithSlope.slope) > 0.66 * abs(avgSlope)) and (abs(lineWithSlope.slope) < 1.33 * abs(avgSlope)):
                prunedLeftLaneLines.append(lineWithSlope)

        # Step 3: right
        totalWeight = 0
        weights = []
        avgSlope = 0
        prunedRightLaneLines = []
        for lineWithSlope in rightLaneLines:
            bottomLine = max(lineWithSlope.y1, lineWithSlope.y2)
            weight = bottomLine / self.bottomY
            totalWeight += weight
            weights.append(weight)
        for idx, lineWithSlope in enumerate(rightLaneLines):
            avgSlope += (lineWithSlope.slope * (weights[idx] / totalWeight))
        # Step 4: right
        for lineWithSlope in rightLaneLines:
            if (abs(lineWithSlope.slope) > 0.66 * abs(avgSlope)) and (abs(lineWithSlope.slope) < 1.33 * abs(avgSlope)):
                prunedRightLaneLines.append(lineWithSlope)

        return prunedLeftLaneLines, prunedRightLaneLines


def calculate_x_on_line(slope, intercept, y):
    return int((y - intercept) / slope)


def calculate_roi_trapezoid(polygonTop, polygonBottom, polygonBottomLeft, polygonBottomRight, polygonLeftSlope, polygonRightSlope):
    """
    Calculates the ROI trapezoid given the top side, the bottom side, the bottom left pixel, the bottom right pixel
    and the slopes of the left and right legs. These could be estimated given the intrinsic and extrinsic camera
    parameters (field of view, focus, zoom, etc.) or estimated through a feedback system.
    :param polygonTop: Top pixel row of the ROI trapezoid
    :param polygonBottom: Bottom pixel row of the ROI trapezoid (could be different than the bottom row of the image
                          when the hood/bonnet is in the image)
    :param polygonBottomLeft: Bottom left pixel of the ROI trapezoid
    :param polygonBottomRight: Bottom right pixel of the ROI trapezoid
    :param polygonLeftSlope: Slope of the left lateral side of the ROI trapezoid
    :param polygonRightSlope: Slope of the right lateral side of the ROI trapezoid
    :return: A numpy array of vertices that constitute the ROI trapezoid
    """
    alt_cs = InvertedYCoordinateSystem(polygonBottom)
    interceptLeft = alt_cs.calculate_intercept(
        slope=polygonLeftSlope,
        x1=polygonBottomLeft,
        y1=polygonBottom)
    interceptRight = alt_cs.calculate_intercept(
        slope=polygonRightSlope,
        x1=polygonBottomRight,
        y1=polygonBottom)
    topLeft = alt_cs.calculate_x_on_line(
        slope=polygonLeftSlope,
        intercept=interceptLeft,
        y=polygonTop)
    topRight = alt_cs.calculate_x_on_line(
        slope=polygonRightSlope,
        intercept=interceptRight,
        y=polygonTop)
    vertices = np.array(
        [[
            [polygonBottomLeft, polygonBottom],
            [topLeft, polygonTop],
            [topRight, polygonTop],
            [polygonBottomRight, polygonBottom]]],
        dtype=np.int32)
    return vertices


def calculate_extrapolated_line(laneLines, topY, bottomY):
    """
    Fit a line (1D) on an array of lines and extrapolate the line to two given Y coordinates
    :param laneLines: The array of lines
    :param topY: The top Y coordinate where the fitted line needs to go through
    :param bottomY: The bottom Y coordinate where the fitted line needs to go through
    :return: A fitted line defined by two points (four coordinates)
    """
    if laneLines:
        x = [line.x1 for line in laneLines]
        x += [line.x2 for line in laneLines]
        y = [line.y1 for line in laneLines]
        y += [line.y2 for line in laneLines]
        coeffs = np.polyfit(x, y, 1)
        x1 = calculate_x_on_line(y=bottomY, slope=coeffs[0], intercept=coeffs[1])
        x2 = calculate_x_on_line(y=topY, slope=coeffs[0], intercept=coeffs[1])
        extrapolatedLine = [x1, bottomY, x2, topY]
        return extrapolatedLine
    else:
        return None


imageFilename = 'solidWhiteCurve.jpg'
image = mpimg.imread(os.path.join('test_images', imageFilename))
polygonTop = 335
polygonBottom = image.shape[0] - 1
polygonBottomLeft = 100
polygonBottomRight = 930
polygonLeftSlope = 0.73
polygonRightSlope = -0.59


# Perform pipeline on all images
plt.interactive(False)
pictureFilenames = os.listdir("test_images/")
vertices = calculate_roi_trapezoid(
    polygonTop=polygonTop,
    polygonBottom=polygonBottom,
    polygonBottomLeft=polygonBottomLeft,
    polygonBottomRight=polygonBottomRight,
    polygonLeftSlope=polygonLeftSlope,
    polygonRightSlope=polygonRightSlope
    )
for picture in pictureFilenames:
    if picture.startswith("output"):
        continue
    image = mpimg.imread(os.path.join('test_images', picture))
    alt_cs = InvertedYCoordinateSystem(polygonBottom)
    grayImage = grayscale(image)
    blurredImage = gaussian_blur(img=grayImage, kernel_size=5)
    cannyEdgeImage = adaptive_canny(image=blurredImage, sigma=0.33)
    maskedImage = region_of_interest(cannyEdgeImage, vertices)
    lineImage, lines = hough_lines(img=maskedImage, rho=1, theta=np.pi/180, min_line_len=20, max_line_gap=20, threshold=15)
    leftLaneLines, rightLaneLines = alt_cs.select_lines(
        lines=lines,
        min_abs_left_slope=0.5,
        min_abs_right_slope=0.5)
    leftLane = calculate_extrapolated_line(leftLaneLines, bottomY=polygonBottom, topY=polygonTop)
    if leftLane:
        draw_lines(lineImage, [[leftLane]], color=[255, 0, 0], thickness=5)
    rightLane = calculate_extrapolated_line(rightLaneLines, bottomY=polygonBottom, topY=polygonTop)
    if rightLane:
        draw_lines(lineImage, [[rightLane]], color=[255, 0, 0], thickness=5)
    blendedImage = weighted_img(img=lineImage, initial_img=image)
    plt.figure()
    plt.imshow(blendedImage, cmap='gray')
    plt.show(block=True)

input("Press Enter to continue...")