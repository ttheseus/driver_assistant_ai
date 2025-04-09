# Model test code

# Imports
import cv2
import torch
from ultralytics import YOLO

# define model
model = YOLO('', task="detect", verbose = False)

# testing stream http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard
video_path = 'http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard'

# classes
class_names = model.names

# define video parameters
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(3)), int(cap.get(4)))
yolo_model_rate = 0.2
analytics_time_payload = 0.035
frame_delay = 1.0/fps if fps > 0 else 1.0 / 30

cv2.namedWindow("YOLO detection", cv2.WINDOW_NORMAL)

# Visualize window and run model
try:
	while cap.isOpened():
		ret, frame = cap.read()
		
		if not ret:
			print("Error, could not read frame")
			break
		
		results = model.track(frame, persist=True)
		annotated_frame = results[0].plot(labels=True, conf=True)
		boxes = results[0].boxes
		
		cv2.imshow("YOLO detection", annotated_frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
except ServiceExit:
	print("Gracefully shutting down...")

cap.release()
cv2.destroyAllWindows()
