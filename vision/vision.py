import cv2 as cv
import numpy as np

class Vision:
  def __init__(self, image, prev_image):
    self.image = image
    self.prev_image = prev_image

  # Returns flow, magnitude, angle
  def computeOpticalFlow(self):
    # Convert both images to gray scale
    img = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
    img_prev = cv.cvtColor(self.prev_image, cv.COLOR_BGR2GRAY)

  def verticalFlow(self, magnitudes):
    pass
  
  def horizontalFlow(self, magnitudes):
    pass

  def expansionFlow(self, flows):
    pass

  def process(self):
    flows, magnitudes = self.computeOpticalFlow()
    left_mag, right_mag = self.horizontalFlow()
    top_mag, bottom_mag = self.verticalFlow()
    expansion = self.expansionFlow()

    