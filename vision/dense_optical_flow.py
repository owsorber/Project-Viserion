import cv2 as cv
import numpy as np


# first frame (make it gray)
img_num=0
img = cv.imread("../images/img" + str(img_num) + ".png", cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_next = cv.imread("../images/img" + str(img_num+1) + ".png", cv.IMREAD_COLOR)
img_next_gray = cv.cvtColor(img_next, cv.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(img)
# Sets image saturation to maximum 
mask[..., 1] = 255

pyr_scale = 0.5
levels = 3
winsize = 5 # was 15
iterations = 3
poly_n = 5
poly_sigma = 1.2
flags = 0
flow = cv.calcOpticalFlowFarneback(img_gray, img_next_gray,  
                                    None, 
                                    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)


# Computes the magnitude and angle of the 2D vectors 
magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
h, w = magnitude.shape

"""
for i in range(h):
  for j in range(w):
    if magnitude[i,j] < 5:
      magnitude[i,j] = 0.0
      angle[i,j] = 0.0
"""

print(magnitude, magnitude.shape)

top_magnitude = np.sum(magnitude[:h//2, :]) / np.sum(magnitude)
bottom_magnitude = np.sum(magnitude[h//2:, :]) / np.sum(magnitude)
left_magnitude = np.sum(magnitude[:, :w//2]) / np.sum(magnitude)
right_magnitude = np.sum(magnitude[:, w//2:]) / np.sum(magnitude)

if top_magnitude > bottom_magnitude:
  print('bottom', bottom_magnitude)
else:
  print('top', top_magnitude)
if left_magnitude > right_magnitude:
  print('right', right_magnitude)
else:
  print('left', left_magnitude)


#print(top_magnitude, bottom_magnitude, left_magnitude, right_magnitude)

# make faster
dot = np.zeros((h,w))
for i in range(h):
  for j in range(w):
    if i == h//2 and j == w//2:
      continue
    from_cam_center = np.array([i-h//2, j-w//2])
    dot[i,j] = np.dot(flow[i,j,:], from_cam_center / np.linalg.norm(from_cam_center))

expansion_magnitude = np.sum(dot)
print(expansion_magnitude)



# Sets image hue according to the optical flow  
# direction 
mask[..., 0] = angle * 180 / np.pi / 2
  
# Sets image value according to the optical flow 
# magnitude (normalized)
mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) 
  
# Converts HSV to RGB (BGR) color representation 
rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

cv.imwrite('../images/img' + str(img_num+1) + '_flow_dense.png', rgb)