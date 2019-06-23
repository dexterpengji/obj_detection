#!/usr/bin/env python
import cv2

face_cascade = cv2.CascadeClassifier('features\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# set resolution
cap.set(3,640)
cap.set(4,480)

while True:
	ret,img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		# PRINT
		print('pisition_x=%s \t pisition_y=%s \t size_w=%s \t size_h=%s' % (x,y,w,h))
	cv2.imshow('img',img)
	if len(faces) == 0:
		print('no face''s been detected')
	if cv2.waitKey(1) &0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
