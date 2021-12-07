#!/usr/bin/env python
import sys
import rospy
import cv2
import math
import numpy as np 
#import python_utils
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler


#display stacked img or only result (=2 or =1)
display = 2

curveList = []
avgVal = 10

widthTop=60
heightTop=160
widthBottom=20
heightBottom=240
wT=480
hT=240
points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

class Test_Img:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image",Image,self.callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    def thresholding(self, img):
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        lowerWhite = np.array([0,0,168])
        upperWhite = np.array([172,111,255])
        maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
        return maskedWhite

    def warpImg (self,img,points,w,h,inv=False):
        pts1 = np.float32(points)
        pts2 = np.float32([(0,0),(w,0),(0,h),(w,h)])
        if inv:
            matrix = cv2.getPerspectiveTransform(pts2,pts1)
        else:
            matrix = cv2.getPerspectiveTransform(pts1,pts2)
        imgWarp = cv2.warpPerspective(img,matrix,(w,h))
        return imgWarp

    def drawPoints(self,img,points):
        for x in range( 0,4):
            cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
        return img

    def getHistogram(self,img,display=False,minPer=0.1,region=1):
        if region == 1:
            histVal = np.sum(img,axis=0)
        else:
            histVal = np.sum(img[int(img.shape[0]//region):,:],axis=0)

        maxVal = np.max(histVal)
        minVal = minPer * maxVal
        indexArray = np.where(histVal >= minVal)
        basePoint = int(np.average(indexArray))

        if display:
            imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
            for x,intensity in enumerate(histVal):
                cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-int(intensity//255//region)),(255,0,255),1)
                cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
            return basePoint,imgHist


    def callback(self,data):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)

        
        cv_image = cv2.resize(cv_image,(480,240))#width,height
        (height,width,channels) = cv_image.shape
        #rospy.loginfo("height=%s, width=%s" % (str(height), str(width)))
        imgResult = cv_image.copy()
        imgCopy = cv_image.copy()
        imgThres = self.thresholding(cv_image)
        imgWarp = self.warpImg(imgThres, points, wT, hT)
        imgWarpPoints = self.drawPoints(imgCopy, points)

        midPoint, imgHist = self.getHistogram(imgWarp,display=True,minPer=0.5,region=4)
        curveAvgPoint, imgHist = self.getHistogram(imgWarp,display=True,minPer=0.9)
        curveRaw = curveAvgPoint - midPoint

        curveList.append(curveRaw)
        if len(curveList) > avgVal:
          curveList.pop(0)
        #FINAL CURVE
        curve = int(sum(curveList)/len(curveList))


        imgInvWarp = self.warpImg(imgWarp, points, wT, hT,inv = True)
        imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:int(hT//3),0:wT] = 0,0,0
        imgLaneColor = np.zeros_like(cv_image)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
        midY = 450
        cv2.putText(imgResult,str(curve),(int(wT//2)-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
        cv2.line(imgResult,(int(wT//2),midY),(int(wT//2)+(curve*3),midY),(255,0,255),5)
        cv2.line(imgResult, ((int(wT // 2) + (curve * 3)), midY-25), (int(wT // 2) + (curve * 3), midY+25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = int(wT // 20)
            cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                        (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        #cv2.putText(imgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3);

        curve = curve/100
        if curve > 1:
            curve = 1
        if curve < -1:
            curve = -1

        self.move(curve)

        if display == 2:
            imgStacked = stackImages(0.7,([cv_image,imgWarpPoints,imgWarp],[imgHist,imgLaneColor,imgResult]))
            cv2.imshow('ImageStack',imgStacked)
        elif display ==1:
            cv2.imshow('Result',imgResult)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()



    def move(self,curveVal):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        
        NegativeCurve = False
        if curveVal>0: NegativeCurve = True

        maxVAl= 1 # MAX CURVE
        if curveVal>maxVAl:curveVal = maxVAl
        elif curveVal<-maxVAl: curveVal =-maxVAl
        else:
            if curveVal<0.2 and curveVal>-0.2: curveVal=0.0   
            elif curveVal<0.4 and curveVal>-0.4: curveVal=0.123    
            elif curveVal<0.5 and curveVal>-0.5: curveVal=0.123    
            elif curveVal<0.6 and curveVal>-0.6: curveVal=0.123    
            elif curveVal<0.7 and curveVal>-0.7: curveVal=0.263    
            elif curveVal<0.8 and curveVal>-0.8: curveVal=0.263     
            elif curveVal<0.9 and curveVal>-0.9: curveVal=0.263 
            elif curveVal<1.0 and curveVal>-1.0: curveVal=0.263    
            elif curveVal<1.2 and curveVal>-1.2: curveVal=0.525 
            elif curveVal<1.5 and curveVal>-1.5: curveVal=0.613 
            else: curveVal=0.349        
        
        if NegativeCurve == True:
            cmd_vel.angular.z = -curveVal #turn right = negative angular
        else:
            cmd_vel.angular.z = curveVal
        #print(curveVal)
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)


if __name__ == '__main__':
    rospy.init_node('line_follower', anonymous=True)
    run = Test_Img()
    rospy.spin()

#Euler Angle to Quaternion
# 1	    0.018
# 2	    0.035
# 3	    0.053
# 4	    0.070
# 5	    0.088
# 6	    0.105
# 7	    0.123
# 8	    0.140
# 9	    0.158
# 10	0.175
# 11	0.193
# 12	0.210
# 13	0.228
# 14	0.245
# 15	0.263
# 16	0.280
# 17	0.298
# 18	0.315
# 19	0.333
# 20	0.350
# 21	0.368
# 22	0.385
# 23	0.403
# 24	0.420
# 25	0.438
# 26	0.455
# 27	0.473
# 28	0.490
# 29	0.508
# 30	0.525
# 31	0.543
# 32	0.560
# 33	0.578
# 34	0.595
# 35	0.613
