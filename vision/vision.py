from simulation.jsbsim_simulator import Simulation
import airsim
import cv2 as cv
import numpy as np
import torch
from math import cos, sin, tan, asin, atan, atan2, sqrt, pi

class Imager:
  def __init__(self, sim: Simulation):
    self.sim = sim
    self.images = []

  def get_np_image(self) -> np.array:
    """
    Gets images from camera '0' as a numpy array
    :return: image_rgb numpy array of with 4 channels of image_type=type
    """
    image_responses = self.sim.client.simGetImages([airsim.ImageRequest('0',
                                                                        airsim.ImageType.Scene,
                                                                        False,
                                                                        False)])
    image_response = image_responses[0]
    image_1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgb = image_1d.reshape(3, image_response.height, image_response.width)
    return image_rgb

  def acquire_image(self):
    img = self.get_np_image
    self.images.append(img)

  def acquired_enough(self):
    return len(self.images) >= 2

  def get_images(self):
    return self.images
  
  def last_two_images(self):
    if not self.acquired_enough():
      raise Exception(f"Imager has only acquired {len(self.iamges)} images")
    return self.images[-1], self.images[-2]
  

class VisionProcessor:
  def __init__(self, image, prev_image, count):
    self.image = image
    self.prev_image = prev_image
    self.count = count
    self.save_flows = True

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
    flow, magnitude, angle = self.compute_optical_flow()
    left_mag, right_mag = self.horizontal_flow(magnitude)
    top_mag, bottom_mag = self.vertical_flow(magnitude)
    expansion = self.expansion_flow()

    if self.save_flows:
      self.visualize(magnitude, angle)

    return (left_mag, right_mag), (top_mag, bottom_mag), expansion, np.sum(magnitude)

  # Write a visualization of the optical flow to a file
  def visualize(self, magnitude, angle):
    # Create a mask image for drawing purposes
    mask = np.zeros_like(magnitude)
    
    # Sets image saturation to maximum 
    mask[..., 1] = 255

    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow 
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) 
    
    # Converts HSV to RGB (BGR) color representation 
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    cv.imwrite('../images/img' + self.count + '.png', rgb)

class VisionGuidanceSystem:
  def __init__(self):
    self.cam_fov = pi
    self.img_h = 144
    self.img_w = 256

    self.side_avoid_threshold = 0.6
    self.expansion_threshold = 0.
    self.ground_dist = 100
  
  # Sets the next waypoint
  def guide(self, vision_processor, autopilot_state):
    # Process the current image with the vision processor
    horiz, vert, expansion, tot_magnitude = vision_processor.process()

    # Decide what pixel we fly toward based on vertical/horizontal flow
    px_i = self.vert_px(vert, expansion)
    px_j = self.horiz_px(horiz, expansion)
    if px_i == None and px_j == None:
      return None # do not set a waypoint, the guidance system is not confident in the need to avoid
    if px_i == None: px_i = int(0.5 * self.img_h)
    if px_j == None: px_j = int(0.5 * self.img_w)
    
    # Acquire relevant current state info
    roll = autopilot_state[4]
    pitch = autopilot_state[5]
    wp = self.ground_dist * torch.Tensor(self.calc_relative_dir_inertial_px(self, roll, pitch, px_i, px_j))

    return wp

  def horiz_px(self, horiz, expansion):
    left, right = horiz
    if left > self.side_avoid_threshold or (left >= 0.5 and expansion > self.expansion_threshold): # avoid the left, go right
      return int(0.75 * self.img_w)
    elif right > self.side_avoid_threshold or (right >= 0.5 and expansion > self.expansion_threshold):
      return int(0.25 * self.img_w) # avoid the right, go left

    return None
  
  def vert_px(self, vert, expansion):
    bottom, top = vert
    if top > self.side_avoid_threshold or (top >= 0.5 and expansion > self.expansion_threshold): # avoid top, go bottom
      return int(0.75 * self.img_h)
    elif bottom > self.side_avoid_threshold or (bottom >= 0.5 and expansion > self.expansion_threshold):
      return int(0.25 * self.img_h) # avoid the right, go left

    return None
  
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
  
  # Calculates the relative direction in converted no-roll/no-pitch reference 
  # frame of the point in an image taken by its camera.
  # px_i = vertical pixel, px_j = horizontal pixel
  # Outputs specifically the ground distance, vertical distance, and relative
  # heading for a unit vector in this direction.
  def calc_relative_dir_inertial_px(self, roll, pitch, px_i, px_j):
    # Get r wrt body frame
    r_body = self.calc_relative_dir_body_px(px_i, px_j)

    # Build DCM
    C_cam = np.matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) # positive 90 degrees wrt c1 pitch to get camera forward
    C_2_roll = np.matrix([[cos(roll), 0, -sin(roll)], [0, 1, 0], [sin(roll), 0, cos(roll)]])
    C_1_pitch = np.matrix([[1, 0, 0], [0, cos(pitch), sin(pitch)], [0, -sin(pitch), cos(pitch)]])
    P_C_I = C_cam @ C_2_roll @ C_1_pitch # only multiply by roll/pitch because we want relative heading still
    I_C_P = np.transpose(P_C_I)

    # Assumes x = right-side, y = forward, z = up
    r_inertial = np.transpose(I_C_P @ r_body)

    # Convert
    ground_distance = np.linalg.norm(r_inertial[:2])
    vertical_distance = r_inertial[2]
    rel_heading = atan2(r_inertial[0], r_inertial[1])

    return np.array([ground_distance, vertical_distance, rel_heading])

