from ultralytics import YOLO 
import cv2 

model = YOLO("yolov8n.pt") 

img = cv2.imread("img4.png") 
results = model(img)


annotated = results[0].plot() 
cv2.imshow("Custom Detection", annotated) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
count = 0

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in vehicle_classes:
            count += 1

print("Vehicle count:", count)