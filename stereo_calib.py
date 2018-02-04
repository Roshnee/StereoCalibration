import numpy as np
import cv2
import glob
import sys

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_left = [] # 2d points in left image plane.
imgpoints_right = [] # 2d points in right image plane
#images = glob.glob('/home/pi/calibration/right/*.jpg')

for i in range(1,11):
#for fname in images:
    img_left = cv2.imread("/home/roshnee/Thesis/rpi code/left/left%02d.jpg"%i)
    img_right= cv2.imread("/home/roshnee/Thesis/rpi code/right/right%02d.jpg"%i)
    gray_left = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
    image_size=gray_left.shape[::-1]
    # Find the chess board corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (9,6),None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (9,6),None)
    
    # If found, add object points, image points (after refining them)
    if ret_left == True and ret_right == True:
        objpoints.append(objp)

        corners2_left = cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
        imgpoints_left.append(corners2_left)
        corners2_right = cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1,-1),criteria)
        imgpoints_right.append(corners2_right)
    
        # Draw and display the corners
        img_left = cv2.drawChessboardCorners(img_left, (9,6), corners2_left,ret_left)
        img_right = cv2.drawChessboardCorners(img_right, (9,6), corners2_right,ret_right)
        cv2.imshow('left img',img_left)
        cv2.imshow('right img',img_right)
        cv2.waitKey(500)
        
ret, mtx_left, dist_left, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1],None,None)
ret, mtx_right, dist_right, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1],None,None)

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
#retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpoints_left,imgpoints_right,image_size)
#retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, (640,720), cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, flags=cv2.cv.CV_CALIB_FIX_INTRINSIC)
#stereocalib_retval, cameraMatrix1, distCoeff1, cameraMatrix2, distCoeff2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,mtx_left,distCoeff1,mtx_right,distCoeff2,gray_right.shape[::-1],criteria=stereocalib_criteria,flags=stereocalib_flags)
termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

print()
print("START - extrinsic calibration ...")
(rms_stereo, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F) = \
cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right,  gray_left.shape[::-1], criteria=termination_criteria_extrinsics, flags=0);
print("camera matrix 1:",cameraMatrix1)
print("CM2:",cameraMatrix2)
print("Rotation Matrix:",R)
print("Transltion:",T)
print("Essential Matrix:",E)
print("Fundamental Matrix:",F)
print("Distortion Coefficients:",distCoeffs1,distCoeffs2)



cv2.destroyAllWindows()
