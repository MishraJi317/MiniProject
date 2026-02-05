from ultralytics import YOLO 
import cv2 

model = YOLO("best_ambulance.pt") 
img = cv2.imread("ambulance2.png") 
results = model(img)
count = 0

for r in results:
    for box in r.boxes:
        count += 1
        
print("Vehicle count:", count)
