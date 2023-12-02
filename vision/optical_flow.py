import cv2 as cv
import numpy as np

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# first frame (make it gray)
img = cv.imread("../images/img0.png", cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_next = cv.imread('../images/img1.png', cv.IMREAD_COLOR)
img_next_gray = cv.cvtColor(img_next, cv.COLOR_BGR2GRAY)

p0 = cv.goodFeaturesToTrack(img_gray, mask = None, **feature_params)

p1, st, err = cv.calcOpticalFlowPyrLK(img_gray, img_next_gray, p0, None, **lk_params)
print('p0', p0)
print('p1', p1)
flow = p1 - p0
print('velocity', flow)

# Create a mask image for drawing purposes
mask = np.zeros_like(img)

color = np.random.randint(0,255,(100,3))
for i,(new,old) in enumerate(zip(p1,p0)):
  a,b = new.ravel()
  c,d = old.ravel()
  mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
  img_next = cv.circle(img_next,(a,b),5,color[i].tolist(),-1)

img_next_flow = cv.add(img_next,mask)

cv.imwrite('../images/img1_flow.png',img_next_flow)