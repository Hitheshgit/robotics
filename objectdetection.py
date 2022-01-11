# import the opencv library
import cv2
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2",threshold = 0.5)
# define a video capture object
cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=328, height=246, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER)

while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = cap.read()
	
	imgCuda = jetson.utils.cudaFromNumpy(frame)
	detections = net.Detect(imgCuda)
	for i in detections:
	    print(i)
	
	img = jetson.utils.cudaToNumpy(imgCuda)

	# Display the resulting frame
	cv2.imshow('frame', img)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

