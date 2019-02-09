import numpy as np
import cv2
import glob
import math as m
import sys

import undisort_images as ui

def main():
	#---------------------- SET THE PARAMETERS
	show_fig = 0
	nRows = 9
	nCols = 6
	# - mm
	dimension = 30

	cam_type = 2

	imageType = '.jpg'
	#------------------------------------------
	print('Camera Calibration')

	if cam_type == 0:
		version, filt = '1', (5,5)
		working_folder = 'Calib_Photos_v'+version +'/'
	elif cam_type == 1:
		version, filt = '640x480', (5,5)
		working_folder = 'Calib_Photos_FE_v'+version +'/'
	elif cam_type == 2:
		version, filt = '752x480', (11,11)
		working_folder = 'Calib_Photos_FE_v'+version +'/'
	


	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, .1)
	# prepare object points
	objp = np.zeros((1,nCols*nRows, 3), np.float32)
	objp[0,:,:2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

	# Arrays to store object points and image points from all the images.
	# 3d point in real world space
	objpoints = []
	# 2d points in image plane.
	imgpoints = []
	# working directory od images
	filename = working_folder + 'im*' + imageType
	print ('Looking in folder ', filename)
	# all images obtention
	images = glob.glob(filename)

	img_bad = None

	print ('Provided images: ', len(images))
	if len(images) < 9:
		print ('Not enough images were found: at least 9 images must be provided')
		sys.exit()

	for fname in images:
		# Read Image
		img = cv2.imread(fname)
		# convert to gray scale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
		# If found, add object points, image points (after refining them)
		if ret:
			objpoints.append(objp)
			corners2 = cv2.cornerSubPix(gray, corners, filt, (-1, -1), criteria)
			imgpoints.append(corners2)
			# Draw and display the corners
			if show_fig:
				img = cv2.drawChessboardCorners(img, (nCols, nRows), corners2, ret)
				cv2.imshow('img', img)
				cv2.waitKey(500)
			if img_bad == None:
				img_bad = fname
		else:
			img_bad = fname

	cv2.destroyAllWindows()

	# read image
	img = cv2.imread(img_bad)
	print('Useful images: ', len(objpoints))
	print('Image dimension: ', img.shape[:2])
	if cam_type == 0:
		mtx, dist, undistoted_img = ui.udisort_normal(objpoints, imgpoints, img)
		path_cam_data = 'Camera_Data/'
	elif cam_type == 1:
		mtx, dist, undistoted_img = ui.undisort_fish_eye(objpoints, imgpoints, img)
		path_cam_data = 'Fisheye_Data/'
	elif cam_type == 2:
		mtx, dist, undistoted_img = ui.undisort_fish_eye_2(objpoints, imgpoints, img)
		path_cam_data = 'Fisheye_Data_2/'


	# save resuts in a .txt file
	filename = path_cam_data+'Camera_Matrix_v'+version+'.txt'
	np.savetxt(filename, mtx, delimiter=',')
	filename = path_cam_data+'Camera_Distortion_v'+version+'.txt'
	np.savetxt(filename, np.transpose(dist), delimiter=',')


	

	cv2.imshow('Undisorted Image', undistoted_img)
	# stop program by typing 'q'
	key = cv2.waitKey(1) & 0xFF
	while key != ord('q'):
		key = cv2.waitKey(1) & 0xFF

	print('End of Camera Calibration')

if __name__ == '__main__':
	main()
