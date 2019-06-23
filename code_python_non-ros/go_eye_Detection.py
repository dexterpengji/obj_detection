import cv2

def detectFace():
	# ubuntu
	features_path = '/run/user/1000/gvfs/smb-share:server=dsm.local,share=private_files/backup_googleDrive/project_TPRobot/packages_python/obj_detection/features/'
	# win
	#features_path = 'features\\'
	face_cascade = cv2.CascadeClassifier(features_path + 'haarcascade_frontalface_default.xml')
	eye_cascade  = cv2.CascadeClassifier(features_path + 'haarcascade_eye.xml')
	eyeC_cascade = cv2.CascadeClassifier(features_path + 'haarcascade_eye_tree_eyeglasses.xml')
	
	if face_cascade.empty() and eye_cascade.empty() and eyeC_cascade.empty():
		raise IOError('Cannot load cascade classifier xml files!')
	cap = cv2.VideoCapture(0)
	scaling_factor = 1.15

	if not cap.isOpened:
		raise IOError('Cannot open webcam!')

	while True:
		ret,frame_raw = cap.read()
		if not ret:
			break
		frame = cv2.resize(frame_raw,None,fx = scaling_factor,fy = scaling_factor,interpolation = cv2.INTER_LINEAR)
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		face_rects = face_cascade.detectMultiScale(gray)

		for (x,y,w,h) in face_rects:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+int(0.6*h),x:x+h]
			roi_color = frame[y:y+int(0.6*h),x:x+h]
			eyes = eye_cascade.detectMultiScale(roi_gray)

			for (x_eye,y_eye,w_eye,h_eye) in eyes:
				cv2.rectangle(frame,(x+x_eye,y+y_eye),(x+x_eye+w_eye,y+y_eye+h_eye),(0,0,255),2)
				roi2_gray = roi_gray[y_eye:y_eye+h_eye,x_eye:x_eye+h_eye]
				#roi2_color = roi_color[y_eye:y_eye+h_eye,x_eye:x_eye+h_eye]
				eyeCs = eyeC_cascade.detectMultiScale(roi2_gray)
				
				for (x_eyeC,y_eyeC,w_eyeC,h_eyeC) in eyeCs:
					center = (int(x_eye+x_eyeC+0.5*w_eyeC),int(y_eye+y_eyeC+0.5*h_eyeC))
					radius = int(0.08*(w_eyeC+h_eyeC))
					color = (0,255,0)
					thickness = 2
					cv2.circle(roi_color,center,radius,color,thickness)

		cv2.imshow('detecting eye',frame)
		#cv2.imshow('frame_raw',frame_raw)

		if cv2.waitKey(1) == 27:	# press ESC to quit
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	detectFace()
