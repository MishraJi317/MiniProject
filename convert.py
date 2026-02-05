from ultralytics import YOLO 
import cv2 

model = YOLO("best_ambulance.pt") 
img = cv2.imread("ambulance2.png") 
results = model(img)

annotated = results[0].plot() 
cv2.imshow("Custom Detection", annotated) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

count = 0

for r in results:
    for box in r.boxes:
        count += 1
        
print("Vehicle count:", count)
