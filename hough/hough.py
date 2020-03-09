import cv2
import numpy as np
from utils import *

class Hough(object):
    kernel_size = 5
    canny_low_threshold = 100
    canny_high_threshold = 200
    mask = [
        [130, 540],         # lower left
        [410, 350],         # upper left
        [570, 350],         # upper right
        [915, 540]          # lower right
    ]
    ignore_mask_color = 255
    # hough parameters
    rho = 1                 # distance resolution of the accumulator in pixels.
    theta = np.pi/180       # angle resolution of the accumulator in radians
    threshold = 30          # minimum number of intersections to “detect” a line
    min_line_len = 20       # minimum line length
    max_line_gap = 20       # maximum allowed gap between points
    # addWeighted parameters
    alpha = 1.0             # weight of the first array elements
    beta = 1.0              # weight of the second array elements
    gamma = 0.0             # scalar added to each sum

    def __init__(self, debug):
        self.debug = debug

    # input should be a canny transformed image
    # output is an image with hough lines drawn
    # see https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
    @staticmethod
    def hough_lines(canny_image, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(canny_image, rho, theta, threshold, np.array([]), 
                minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((*canny_image.shape, 3), dtype=np.uint8)
        Hough.draw_lines(line_img, lines)
        return line_img

    @staticmethod
    def draw_lines(canny_image, lines, color=[255, 0, 0], thickness=2):
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(canny_image, (x1, y1), (x2, y2), color, thickness)
    
    def process(self, image):
        # remove color, go from 3 channels (RGB) to 1 channel
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.debug: show(grayscale, 'hough - grayscale')
        # apply a low-pass filter kernel, useful for removing noise
        # removes high frequency content - noise, edges - from the image
        gaussian_blur = cv2.GaussianBlur(grayscale, (self.kernel_size, self.kernel_size), 0)
        if self.debug: show(gaussian_blur, 'hough - gaussian blur')
        # find edges in the input image and marks them in the output map edges 
        # using the Canny algorithm
        canny_image = cv2.Canny(gaussian_blur, self.canny_low_threshold, self.canny_high_threshold)
        if self.debug: show(canny_image, 'hough - canny transform')
        # apply mask
        points = np.array([self.mask], dtype=np.int32)
        # define a blank mask
        mask = np.zeros_like(canny_image)        
        # fill pixels inside the polygon defined by points with the fill color    
        cv2.fillPoly(mask, points, self.ignore_mask_color)
        # uncomment next line to see the masked area
        # canny_image.fill(255)
        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(canny_image, mask)
        if self.debug: show(masked_image, 'hough - masked image')
        hough_image = Hough.hough_lines(masked_image, self.rho, self.theta, 
                  self.threshold, self.min_line_len, self.max_line_gap)
        if self.debug: show(hough_image, 'hough - hough image')
        # image is the original image, hough_image is blank with hough lines on it
        final_image = cv2.addWeighted(image, self.alpha, hough_image, self.beta, self.gamma)
        if self.debug: show(final_image, 'hough - final image')

        return final_image