import numpy as np
import cv2
import imutils
import pyautogui as gui
from collections import deque
gui.PAUSE = 0
gui.FAILSAFE=False

# define the lower and upper boundaries of the "green" ball in the HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

pts = deque(maxlen=32) #Making list of tracked Points
counter = 0 #Frame Couter
(dX, dY) = (0, 0) #relative Position
direction = ""
vs = cv2.VideoCapture(0)

while True:
        ret,frame = vs.read()
        frame=cv2.flip(frame,1)#FlippinTheFrame
        
        #  blur  the Frame, and convert it to the HSV color space
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # make a mask for the Range of Green, then removing noise left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current  x, y centeroid of the ball
        cnts= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        centeroid = None

       
        if len(cnts) > 0:  # if at least one contror was found
                c = max(cnts, key=cv2.contourArea) # find the largest contour Area
                ((x, y), radius) = cv2.minEnclosingCircle(c) #&  fit that area in  circle and centroid
                M = cv2.moments(c)
                centeroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #(Centerr=cx/cy)

                # if the radius is minimum 10 then draw
                if radius > 10:
                        cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)  # draw the circle
                        cv2.circle(frame, centeroid, 5, (0, 0, 255), -1) # Drawing centroid
                        pts.appendleft(centeroid)  # then update the list of tracked points

        
        for i in np.arange(1, len(pts)):   #if ball is not detected then Loop and tracked POints are not found then ignore them
                if pts[i - 1] is None or pts[i] is None:
                        continue

                # check to see if enough points have been in  the buffer
                if counter >= 10 and i == 10 and pts[-10] is not None:
                        # compute the difference between the x and y and set direction
                        dX = pts[i-10][0] - pts[i][0]
                        dY = pts[i-10][1] - pts[i][1]
                        (dirX, dirY) = ("", "")

                        # if there is Horizontal MOvement
                        if np.abs(dX) > 20:
                                dirX = "Right" if np.sign(dX) == 1 else "Left"

                        #if there is Vertical Movement
                        if np.abs(dY) > 20:
                                dirY = "Bottom" if np.sign(dY) == 1 else "Top"

                        # handle when both directions are non-empty
                        if dirX != "" and dirY != "":
                                direction = "{}-{}".format(dirY, dirX)

                        # otherwise, only one direction is non-empty
                        else:
                                direction = dirX if dirX != "" else dirY

                cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness=4)

        # show the movement deltas and the direction of movement on the frame
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 255), 3)
        gui.moveRel(dX,dY,duration=0)
        '''if  dX==0 and dY==0:
                gui.doubleClick()'''

        # show the frame to our screen and increment the frame counter
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key==27:     break
        counter += 1
vs.release()
cv2.destroyAllWindows() #Close all windows
