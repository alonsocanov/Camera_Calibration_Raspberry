import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'

import numpy as np
import os
import glob

CHECKERBOARD = (6,9)
DIMENSION = 30

working_folder = 'Calib_Photos_FE_v640x480/'

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, DIMENSION, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(working_folder+'im*.jpg')

for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)

N_OK = len(objpoints)
mtx = np.zeros((3, 3))
dist = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],mtx,dist,rvecs,tvecs,calibration_flags,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

print("Found ",N_OK, " valid images for calibration")
print('Image dimension', _img_shape[::-1])
print('Calibration Matrix')
print(mtx)
print('Distortion')
print(np.transpose(dist))


DIM=_img_shape[::-1]

img_path = working_folder+'im1.jpg'
img = cv2.imread(img_path)
h,w = img.shape[:2]

map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, DIM, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()