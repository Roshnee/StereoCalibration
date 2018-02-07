import numpy as np
import cv2
import glob
import sys
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
cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right,  gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC, criteria=termination_criteria_extrinsics);
print("camera matrix 1:",cameraMatrix1)
print("CM2:",cameraMatrix2)
print("Rotation Matrix:",R)
print("Transltion:",T)
Baseline=2.4*np.linalg.norm(T)
print("Baseline:",Baseline)
print("Essential Matrix:",E)
print("Fundamental Matrix:",F)
print("Distortion Coefficients:",distCoeffs1,distCoeffs2)

rectify_scale = 0 # 0=full crop, 1=no crop
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (640, 720), R, T, alpha = 0)
mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (640, 720), cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (640, 720), cv2.CV_32FC1)

#lFrame = cv2.imread('/home/roshnee/Thesis/rpi code/left/left01.jpg')
#rFrame = cv2.imread('/home/roshnee/Thesis/rpi code/right/right01.jpg')

#Input from the video stream
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
       cv2.imshow('Disparity Map', filteredImg)
       


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()






#for line in range(0, int(right_img_remap.shape[0] / 20)):
#    left_img_remap[line * 20, :] = (0, 0, 255)
#    right_img_remap[line * 20, :] = (0, 0, 255)
#cv2.imshow('rectified left',left_img_remap)
#cv2.imshow('rectified right',right_img_remap)
#cv2.imshow('winname', np.hstack([left_img_remap, right_img_remap]))
# Assuming you have left01.jpg and right01.jpg that you want to rectify
#lFrame = cv2.imread('/home/roshnee/Thesis/rpi code/left/left01.jpg')
#rFrame = cv2.imread('/home/roshnee/Thesis/rpi code/right/right01.jpg')
#w, h = lFrame.shape[:2] # both frames should be of same shape
#frames = [lFrame, rFrame]
# Params from camera calibration
#camMats = [cameraMatrix1, cameraMatrix2]
#distCoeffs = [distCoeffs1, distCoeffs2]
#camSources = [0,1]
#for src in camSources:
#    distCoeffs[src][0][4] = 0.0 # use only the first 2 values in distCoeffs
# The rectification process
#newCams = [0,0]
#roi = [0,0]
#for src in camSources:
#    newCams[src], roi[src] = cv2.getOptimalNewCameraMatrix(cameraMatrix = camMats[src], 
#                                                           distCoeffs = distCoeffs[src], 
#                                                           imageSize = (w,h), 
#                                                           alpha = 0)
#rectFrames = [0,0]
#for src in camSources:
#        rectFrames[src] = cv2.undistort(frames[src], 
#                                        camMats[src], 
#                                        distCoeffs[src])
# See the results
#view = np.hstack([frames[0], frames[1]])    
#rectView = np.hstack([rectFrames[0], rectFrames[1]])

#cv2.imshow('view', view)
#cv2.imshow('rectView', rectView)

# Wait indefinitely for any keypress

