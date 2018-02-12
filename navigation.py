#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import numpy as np
import cv2
import glob
import sys
import math
import time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_left = [] # 2d points in left image plane.
imgpoints_right = [] # 2d points in right image plane

for i in range(1,22):
    img_left = cv2.imread("/home/roshnee/myrepo/StereoCalibration/left/left%02d.jpg"%i)
    img_right= cv2.imread("/home/roshnee/myrepo/StereoCalibration/right/right%02d.jpg"%i)

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

#calibrate the cameras separately to get the camera matrices and distortion coefficients        
ret, mtx_left, dist_left, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1],None,None)
ret, mtx_right, dist_right, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1],None,None)

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_FIX_INTRINSIC
print("Dist_left:",dist_left)
print("Dist_right:",dist_right)
termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

print("START - extrinsic calibration ...")

#calibrate both the cameras
(rms_stereo, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F) = \
cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx_left, 0, mtx_right, 0,  gray_left.shape[::-1],flags=cv2.CALIB_FIX_INTRINSIC, criteria=termination_criteria_extrinsics)
print("RMS:",rms_stereo)
print("camera matrix 1:",cameraMatrix1)
print("CM2:",cameraMatrix2)
print("Rotation Matrix:",R)
print("Translation:",T)
print("Essential Matrix:",E)
print("Fundamental Matrix:",F)
print("Distortion Coefficients:",distCoeffs1,distCoeffs2)
Baseline=2.4*np.linalg.norm(T)
print("baseline in cm",Baseline)
rectify_scale = 0 # 0=full crop, 1=no crop, -1=default without scaling

#Rectify to get the distortion coefficients and the Q matrix needed for reprojection
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (640, 640), R, T,
                                  flags=cv2.CALIB_ZERO_DISPARITY, alpha = 0)

print('R1:',R1)
print('R2:',R2)
print('P1:',P1)
print('P2',P2)
print('Q matrix:',Q)

mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (640, 640), cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (640, 640), cv2.CV_32FC1)

#connect to rpi
HOST ="192.168.0.100" # Enter IP or Hostname of your server
PORT = 9900 # Pick an open Port (1000+ recommended), must match the server port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))


#Input from the video stream
nof=0
time.sleep(2)
cap = cv2.VideoCapture('http://pi:raspberrypi@192.168.0.100:9090/stream/video.mjpeg')
print("Cap is opened")
while(cap.isOpened()):

    ret,frame=cap.read()
    if ret==True:
       cv2.imshow('video stream',frame)
       right=frame[0:640,0:640]
       left=frame[0:640,640:1280]
       left_img_remap = cv2.remap(left, mapL1, mapL2, cv2.INTER_LINEAR)
       right_img_remap = cv2.remap(right, mapR1, mapR2, cv2.INTER_LINEAR)
       cv2.imshow('rectified left',left_img_remap)
       cv2.imshow('rectified right',right_img_remap)
     
       #Convert to grayscale
       imgL=cv2.cvtColor(left_img_remap,cv2.COLOR_BGR2GRAY)
       imgR=cv2.cvtColor(right_img_remap,cv2.COLOR_BGR2GRAY)
       #find the depth map
       # SGBM Parameters -----------------
       window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
       left_matcher = cv2.StereoSGBM_create(
    	  	minDisparity=0,
    		numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    		blockSize=5,
    		P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    		P2=32 * 3 * window_size ** 2,
    		disp12MaxDiff=1,
    		uniquenessRatio=15,
    		speckleWindowSize=0,
    		speckleRange=2,
    		preFilterCap=63,
    		mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
       )

       right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
 
       # FILTER Parameters
       lmbda = 80000
       sigma = 1.2
       visual_multiplier = 1.0
       wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
       wls_filter.setLambda(lmbda)
       wls_filter.setSigmaColor(sigma)
 
       print('computing disparity...')
       displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
       dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
       displ = np.int16(displ)
       dispr = np.int16(dispr)
       filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
 
       filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
       filteredImg = np.uint8(filteredImg)
       filteredImg=filteredImg[0:640,100:640]
       nop=345600
       temp=0
       tri_const=650*0.065
       #Z=tri_const/filteredImg[320,320]
       #print("Z=",Z,"for the disparity value",filteredImg[320,320])
       for i in range(0,640):
         for j in range(0,540):
            temp=temp+filteredImg[i,j]
       avg=temp/nop
       Z_avg=tri_const/avg
       print("average",avg)
       print("Average depth",Z_avg)
       cv2.imshow('Disparity Map', filteredImg)
       nof=nof+1
      # if(nof%20 == 0):

        # if(Z_avg>0.3):
        #      s.send("forward".encode())
        #      reply = s.recv(1024)
        #      print(reply)
        # elif(Z_avg<0.3):
        #     s.send("right".encode())
        #     reply = s.recv(1024)
        #     print(reply)
        # else:
        #    s.send("quit".encode())
        #    reply = s.recv(1024)
        #    print(reply)
       #else:
       #  continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
         #s.send("quit".encode())
         #reply = s.recv(1024)
         #print(reply)
         break

cv2.waitKey(0)
cv2.destroyAllWindows()


 















#lFrame = cv2.imread('/home/roshnee/myrepo/StereoCalibration/left/left01.jpg')
#rFrame = cv2.imread('/home/roshnee/myrepo/StereoCalibration/right/right01.jpg')
 
#grayL = cv2.cvtColor(lFrame,cv2.COLOR_BGR2GRAY);
#grayR = cv2.cvtColor(rFrame,cv2.COLOR_BGR2GRAY);

#undistort the image and remap
#undistorted_rectifiedL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LANCZOS4);
#undistorted_rectifiedR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LANCZOS4);


#for line in range(0, 32):
#    undistorted_rectifiedL[line * 20, :] = (0, 0, 255)
#    undistorted_rectifiedR[line * 20, :] = (0, 0, 255)

#cv2.imshow('rectified left',undistorted_rectifiedL)
#cv2.imshow('rectified right',undistorted_rectifiedR)





















#cv2.imshow('winname', np.hstack([undistorted_rectifiedL, undistorted_rectifiedR]))
#cv2.waitKey(0)
#exit(0)


# display image
#cv2.imshow('windowNameL',undistorted_rectifiedL);
#cv2.imshow('windowNameR',undistorted_rectifiedR);


#Find the correspondent points and the depth map
#computeDepthMap=cv2.StereoBM_create(numDisparities=128, blockSize=21)
#disparity=computeDepthMap.compute(undistorted_rectifiedL,undistorted_rectifiedR)

# scale the disparity to 8-bit for viewing

#depthmap = (disparity / 16.).astype(np.uint8) + abs(disparity.min())

#cv2.imshow('depthmap',depthmap)

#for i in range(100):
#  for j in range(100):
#    print("Depthmap[",i,j,"]",depthmap[i,j])


#sys.stdout=open('point cloud.txt','w')
#ddepth=cv2.reprojectImageTo3D(depthmap,Q,_3dImage,handleMissingValues=0)
#sys.stdout.close()

#cv2.waitKey(0)
#cv2.destroyAllWindows()
