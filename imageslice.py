import cv2
import glob
#import Image
import itertools

images=glob.glob("/home/roshnee/PycharmProjects/*.jpg")

i=1
for im in images:
    #image=Image.open(im)
    #image.show()
    image=cv2.imread(im)
    cv2.imshow('image',image)
    right=image[0:640,0:640]
    left=image[0:640,640:1280]
    cv2.imwrite('/home/roshnee/myrepo/StereoCalibration/left/left%02d.jpg'%i,left)
    cv2.imwrite('/home/roshnee/myrepo/StereoCalibration/right/right%02d.jpg'%i,right)
    i=i+1
    cv2.waitKey(500)
    

cv2.destroyAllWindows()
