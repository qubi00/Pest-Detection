import time
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt # Still useful for plotting metrics later

model = YOLO("best6.pt")
model.task = 'detect'

# --- 1. CHANGE SOURCE TO VIDEO FILE ---
source = "test.mp4" 
output_path = "output_test_annotated.mp4" # Define the output file name

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"Error: Could not open video file: {source}")
    exit()

# --- 2. SETUP VIDEO WRITER ---
# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec (MP4V is widely compatible) and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {source} (FPS: {fps:.2f}, Resolution: {frame_width}x{frame_height})")

frames = 0
inference_times = []
total_start = time.time()

while True:
    success, frame = cap.read()
    
    if not success:
        print("Video processing finished.")
        break
    
    frames += 1

    start = time.time() 
    # Use 'cpu' device for cluster processing unless you are sure of GPU availability
    results = model(frame, conf=0.5) 
    end = time.time() 
    inference_time = (end - start) * 1000 
    inference_times.append(inference_time) 

    # --- DETECTION AND ANNOTATION (REMAINS THE SAME) ---
    for result in results:
        # Ultralytics provides a built-in plot function, but we'll stick to your custom OpenCV drawing
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = int(box.conf[0] * 100)
            label = f"Mosquito {conf}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # --- 3. WRITE THE ANNOTATED FRAME TO THE OUTPUT FILE ---
    out.write(frame)
    
    # Optional: Print progress every 100 frames
    if frames % 100 == 0:
        print(f"Processed {frames} frames...")


total_end = time.time()
total_elapsed = total_end - total_start

# --- RELEASE EVERYTHING ---
cap.release()
out.release() 
cv2.destroyAllWindows()

# --- PRINT METRICS ---
if inference_times:
    inference_times = inference_times[1:]
    # Plotting might not display on SCC OnDemand, but the file will be generated
    plt.figure(figsize=(12, 6))
    plt.plot(inference_times, label="Inference time (ms)", color='blue')
    plt.xlabel("Frame number")
    plt.ylabel("Inference time (ms)")
    plt.title("YOLO Inference Time Per Frame")
    plt.grid(True)
    plt.legend(fontsize=24)
    plt.savefig("inference_time_plot.png") # Save plot to file instead of showing
    
    print("\n--- Performance Metrics ---")
    print(f"Average Inference: {sum(inference_times)/len(inference_times):.2f} ms") 
    print(f"Worst-case Inference: {max(inference_times):.2f} ms") 
    print(f"Total time elapsed: {(total_elapsed):.2f} s")
    print(f"Total frames processed: {frames}")
    print(f"Camera FPS (Source Video FPS): {(fps):.2f}")
    print(f"Actual Processing FPS: {(frames / (total_elapsed)):.2f}")

print(f"\nSuccessfully saved annotated video to: {output_path}")