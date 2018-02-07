import numpy as np
import cv2
import time 

time.sleep(2)
cap = cv2.VideoCapture('http://pi:raspberrypi@192.168.0.100:9090/stream/video.mjpeg')

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')

#out = cv2.VideoWriter('http://192.168.0.100:9090/stream/video.mpeg',fourcc, 20.0, (640,480))


#cap=cv2.VideoCapture('http://192.168.0.100:9090/stream/video.mpeg')

while(cap.isOpened()):

    print("Cap is open")
    ret,frame=cap.read()
    if ret==True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
