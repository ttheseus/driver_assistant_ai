from ultralytics import YOLO 

# Load a model
model = YOLO("yolov8n.pt") # build model from scratch

results = model.train(data="config.yaml", epochs=295) #train the model