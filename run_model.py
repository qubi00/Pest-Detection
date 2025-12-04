import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
model.task = 'detect'

# source 0 is webcam
source = 0

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Press 'q' to quit.")

while True:
    success, frame = cap.read()
    
    if not success:
        print("Video finished or failed to read.")
        break

    # Only show if at least 50% certain
    results = model(frame, conf=0.5)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            conf = int(box.conf[0] * 100)
            
            label = f"Mosquito {conf}%"

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw Label Background 
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1)

            cv2.putText(frame, label, (x1, y1 - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("Mosquito Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()