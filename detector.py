from ultralytics import YOLO
import cv2
from concurrent.futures import ThreadPoolExecutor

# Load models ONCE (very important for speed)
ambulance_model = YOLO("best_ambulance.pt")
vehicle_model = YOLO("yolov8n.pt")

img = cv2.imread("ambulance2.png")

vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']


def run_ambulance_model():
    results = ambulance_model(img)
    count = 0
    for r in results:
        count += len(r.boxes)
    return count


def run_vehicle_model():
    results = vehicle_model(img)
    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = vehicle_model.names[cls]
            if label in vehicle_classes:
                count += 1
    return count


# Run both models in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(run_ambulance_model)
    future2 = executor.submit(run_vehicle_model)

    ambulance_count = future1.result()
    vehicle_count = future2.result()

print("Ambulance count:", ambulance_count)
print("Vehicle count:", vehicle_count)
