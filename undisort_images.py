# undisort images
import numpy as np
import cv2
import glob
import math as m


def udisort_normal(objpoints, imgpoints, img):
	if len(objpoints) > 1:
		h,  w = img.shape[:2]
		# Undistort an image
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
		#cv2.imshow('Disorted Image', img)
		#cv2.waitKey(500)
		
		new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
		undisorted_img = cv2.undistort(img, mtx, dist, None, new_mtx)
		# Crop image
		if roi[0]:
			x,y,w,h = roi
			undisorted_img = undisorted_img[y:y+h, x:x+w]

		print ('ROI: ', roi)
		print('Calibration Matrix: ')
		print(mtx)
		print('New Camera Martrix:')
		print(new_mtx)
		print('Disortion: ')
		print(dist)

		# Distortion Error
		mean_error = 0
		for i in range(len(objpoints)):
			imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
			error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			mean_error += error

		print('Total error: ', mean_error/len(objpoints))
	else:
		print('Not enough images')
		mtx, dist, rvecs, tvecs, undisorted_img = 0,0,0,0,0

	return mtx, dist, undisorted_img

# first version of fish eye disortion
def undisort_fish_eye(objpoints, imgpoints, img):
	calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
	criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

	nb_img = len(objpoints)
	h,  w = img.shape[:2]
	mtx = np.zeros((3, 3))
	dist = np.zeros((4, 1))
	rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(nb_img)]
	tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(nb_img)]
	rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints,imgpoints,(w, h),mtx,dist,rvecs,tvecs,calibration_flags, criteria)

	map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (w, h), cv2.CV_16SC2)
	undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	print('Calibration Matrix: ')
	print(mtx)
	print('Disortion: ')
	print(np.transpose(dist))

	return mtx, dist, undistorted_img

# crop image
def crop(img, h_dim, w_dim):
	w, h, l = img.shape
	img = img[h_dim[0]:h_dim[1],w_dim[0]:w_dim[1],:l]
	Resize the image
	img = cv2.resize(img, (h, w))

	
