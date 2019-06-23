#!/usr/bin/env python
import roslib
import sys
import rospy
import time
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import String

path_features = '/home/pengji/catkin_ws/src/obj_detection/scripts/features/'
face_cascade = cv2.CascadeClassifier(path_features+'haarcascade_frontalface_default.xml')
eye_cascade =  cv2.CascadeClassifier(path_features+'haarcascade_eye.xml')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

class FaceTracking(object):
	def __init__(self):
		self.bridge_object = CvBridge()
		self.image_sub = rospy.Subscriber("usb_cam/image_raw", Image, self.camera_callback)
	
	def camera_callback(self, data):
		try:
			cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
		except CvBridgeError as e:
			print(e)
		gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		#found,w=hog.detectMultiScale(cv_image, winStride=(8,8), padding=(32,32), scale=1.05)
		#draw_detections(cv_image,found)
		for (x,y,w,h) in faces:
			cv2.rectangle(cv_image,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = cv_image[y:y+h, x:x+w]
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				#info_eyes = (ex,ey,ew,eh,rospy.get_time())
		cv2.imshow("Image window - cv_image", cv_image)
		#cv2.imshow("Image window - gray", gray)
		#rospy.loginfo(info)
		cv2.waitKey(1)

def main():
	faceTracking_object = FaceTracking()
	rospy.init_node('face_tracking_node', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down by keyboard")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
