import numpy as np
import cv2
import glob
import math as m

#---------------------- SET THE PARAMETERS
nRows = 9
nCols = 6
# - mm
dimension = 25
#version, filt = '720x480', (5, 5)
version, filt = '1', (5, 5)
workingFolder = 'Calib_Photos_v'+version +'/'
#workingFolder = 'Calib_Photos_FE_v'+version +'/'

imageType = '.jpg'
#------------------------------------------
print('Camera Calibration')
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nCols*nRows, 3), np.float32)
objp[:, : 2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
# 3d point in real world space
objpoints = []
# 2d points in image plane.
imgpoints = []
# working directory od images
filename = workingFolder + 'im*' + imageType
print ('Looking in folder ', filename)
# all images obtention
images = glob.glob(filename)

imgNotGood = None

print ('Number of images provided: ', len(images))
if len(images) < 9:
    print ('Not enough images were found: at least 9 images must be provided')
    sys.exit()

for fname in images:
	# Read Image
    img = cv2.imread(fname)
    # Image dimentions 
    w, h, l = img.shape
    # Crop Image
    img = img[50:h,150:w,:l]
    # Resize the image
    img = cv2.resize(img, (h, w))
    # cv2.imshow('img',img)
    # cv2.waitKey(500)
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, filt, (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (nCols, nRows), corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
        if imgNotGood == None:
            imgNotGood = fname
    else:
        imgNotGood = fname

print('Number of useful images: ', len(objpoints))
cv2.destroyAllWindows()

if len(objpoints) > 1:
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    # Undistort an image
    img = cv2.imread(imgNotGood)
    #cv2.imshow('Disorted Image', img)
    #cv2.waitKey(500)
    h,  w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    undistoted_img = cv2.undistort(img, mtx, dist, None, new_mtx)
    # Crop image
    x,y,w,h = roi
    undistoted_img = undistoted_img[y:y+h, x:x+w]
    print ('ROI: ', roi)
    print('Calibration Matrix: ')
    print(mtx)
    print('New Camera Martrix:')
    print(new_mtx)
    print('Disortion: ')
    print(dist)

    #--------- Save result
    filename = 'Camera_Data/Camera_Matrix_v'+version+'.txt'
    np.savetxt(filename, mtx, delimiter=',')
    filename = 'Camera_Data/Camera_new_Matrix_v'+version+'.txt'
    np.savetxt(filename, new_mtx, delimiter=',')
    filename = 'Camera_Data/Camera_Distortion_v'+version+'.txt'
    np.savetxt(filename, dist, delimiter=',')

    # Distortion Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print('Total Error: ', mean_error/len(objpoints))

    cv2.imshow('Undisorted Image', undistoted_img)
    key = cv2.waitKey(1) & 0xFF
    while key != ord('q'):
        key = cv2.waitKey(1) & 0xFF

else:
    print('In order to calibrate you need at least 9 good pictures... try again')

print('End of Camera Calibration')
