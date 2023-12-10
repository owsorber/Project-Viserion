import cv2 as cv
import numpy as np
from math import cos, sin, tan, asin, atan, atan2, sqrt, pi

class VisionProcessor:
  def __init__(self, image, prev_image):
    self.image = image
    self.prev_image = prev_image

  # Returns flow, magnitude, angle
  def compute_optical_flow(self):
    # Convert both images to gray scale
    img = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
    img_prev = cv.cvtColor(self.prev_image, cv.COLOR_BGR2GRAY)

    # Set parameters and run optical flow
    pyr_scale = 0.5
    levels = 3
    winsize = 5 # was 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    flags = 0
    flow = cv.calcOpticalFlowFarneback(img_prev, img,  
                                        None, 
                                        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    return flow, magnitude, angle

  def vertical_flow(self, magnitude):
    h, _ = magnitude.shape
    top_magnitude = np.sum(magnitude[:h//2, :]) / np.sum(magnitude)
    bottom_magnitude = np.sum(magnitude[h//2:, :]) / np.sum(magnitude)
    return top_magnitude, bottom_magnitude
  
  def horizontal_flow(self, magnitude):
    _, w = magnitude.shape
    left_magnitude = np.sum(magnitude[:, :w//2]) / np.sum(magnitude)
    right_magnitude = np.sum(magnitude[:, w//2:]) / np.sum(magnitude)
    return left_magnitude, right_magnitude

  def expansion_flow(self, flow):
    h, w = flow.shape

    # This is done with for loops and is slow. Should consider attempting to vectorize
    dot = np.zeros((h,w))
    for i in range(h):
      for j in range(w):
        if i == h//2 and j == w//2:
          continue
        from_cam_center = np.array([i-h//2, j-w//2])
        dot[i,j] = np.dot(flow[i,j,:], from_cam_center / np.linalg.norm(from_cam_center))

    return np.sum(dot)

  # Returns horizontal flow, vertical flow, expansion flow
  def process(self):
    flows, magnitudes = self.compute_optical_flow()
    left_mag, right_mag = self.horizontal_flow()
    top_mag, bottom_mag = self.vertical_flow()
    expansion = self.expansion_flow()
    return (left_mag, right_mag), (top_mag, bottom_mag), expansion 


class VisionGuidanceSystem:
  def __init__(self):
    self.cam_fov = pi
    self.img_h = 144
    self.img_w = 256

    self.side_avoid_threshold = 0.6
  
  # Sets the next waypoint
  def guide(self, vision_processor):
    horiz, vert, expansion = vision_processor.process()
    


  def horiz_px(self, horiz):
    left, right = horiz
    if left > self.side_avoid_threshold: # avoid the left, go right
      return int(0.75 * self.img_w)
    elif right > self.side_avoid_threshold:
      return int(0.25 * self.img_w) # avoid the right, go left

    return int(0.5 * self.img_w)
  
  def vert_px(self, vert):
    bottom, top = vert
    if top > self.side_avoid_threshold: # top, go bottom
      return int(0.75 * self.img_h)
    elif bottom > self.side_avoid_threshold:
      return int(0.25 * self.img_h) # avoid the right, go left

    return int(0.5 * self.img_h)
  
  # Calculates the relative direction wrt plane body frame of the point in an
  # image taken by its camera; px_i = vertical pixel, px_j = horizontal pixel
  def calc_relative_dir_body_px(self, px_i, px_j):
    # Normalize pixels to proportion from center -> [-0.5, 0.5]
    px_i = (px_i - self.img_h / 2) / self.img_h
    px_j = (px_j - self.img_w / 2) / self.img_w

    # Calculate angles in each direction
    vert_angle = px_i * self.cam_fov
    horiz_angle = px_j * self.cam_fov

    # Calculate unit vector
    dir = np.array([np.sin(vert_angle), np.sin(horiz_angle)])
    norm = np.linalg.norm(dir)
    return dir / norm if norm != 0 else dir
  
  # Calculates the relative direction in inertial frame of the point in an
  # image taken by its camera; px_i = vertical pixel, px_j = horizontal pixel
  def calc_relative_dir_inertial_px(self, roll, pitch, yaw, px_i, px_j):
    # Get r wrt body frame
    r_body = self.calc_relative_dir_body_px(px_i, px_j)

    # Build DCM
    C_cam = np.matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) # positive 90 degrees wrt c1 pitch to get camera forward
    C_2_roll = np.matrix([[cos(roll), 0, -sin(roll)], [0, 1, 0], [sin(roll), 0, cos(roll)]])
    C_1_pitch = np.matrix([[1, 0, 0], [0, cos(pitch), sin(pitch)], [0, -sin(pitch), cos(pitch)]])
    C_3_yaw = np.matrix([[cos(-yaw), sin(-yaw), 0], [-sin(-yaw), cos(-yaw), 0], [0, 0, 1]])
    P_C_I = C_cam @ C_2_roll @ C_1_pitch @ C_3_yaw
    I_C_P = np.transpose(P_C_I)

    # Assumes x = east, y = north, z = up
    r_inertial = np.transpose(I_C_P @ r_body)

    # Convert
    x, y, z = r_inertial[0], r_inertial[1], r_inertial[2]
    return np.array([y, x, z])
