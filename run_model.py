import time
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("best.pt")
model.task = 'detect'

# source 0 is webcam
source = 0

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Press 'q' to quit.")

inference_times = []
total_start = time.time()

while True:
    success, frame = cap.read()
    
    if not success:
        print("Video finished or failed to read.")
        break

    start = time.time()
    results = model(frame, conf=0.5)
    end = time.time()

    inference_time = (end - start) * 1000
    inference_times.append(inference_time)

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

total_end = time.time()
total_elapsed = total_end - total_start

if inference_times:
    inference_times = inference_times[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(inference_times, label="Inference time (ms)", color='blue')
    plt.xlabel("Frame number")
    plt.ylabel("Inference time (ms)")
    plt.title("YOLO Inference Time Per Frame")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Average: {sum(inference_times)/len(inference_times):.2f} ms")
    print(f"Worst-case: {max(inference_times):.2f} ms")
    print(f"Total time elapsed: {(total_elapsed):.2f} ms")
    print(f"Total frames: {len(inference_times)}")
