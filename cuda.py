import time
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("best7.pt")
model.task = 'detect'

# --- 1. CHANGE SOURCE TO VIDEO FILE ---
source = "test.mp4" 
output_path = "output_test_annotated.mp4"

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"Error: Could not open video file: {source}")
    exit()

# --- 2. SETUP VIDEO WRITER ---
# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {source} (FPS: {fps:.2f}, Resolution: {frame_width}x{frame_height})")

frames = 0
inference_times = []
total_start = time.time()

read_times = []
plot_times = []
write_times = []

while True:
    read_start = time.time()
    success, frame = cap.read()
    read_end = time.time()    

    if not success:
        print("Video processing finished.")
        break
    
    frames += 1
    read_time = (read_end - read_start) * 1000
    read_times.append(read_time)

    if frames % 2 == 0:
        continue

    start = time.time() 
    # Use 'cpu' device for cluster processing unless you are sure of GPU availability
    results = model(frame, conf=0.5, device='cuda:2', verbose=False) 
    end = time.time() 
    inference_time = (end - start) * 1000 
    inference_times.append(inference_time) 

    plot_start = time.time()
    frame = results[0].plot()
    plot_end = time.time()
    plot_time = (plot_end - plot_start) * 1000
    plot_times.append(plot_time)

    write_start = time.time()
    out.write(frame)
    write_end = time.time()
    write_time = (write_end - write_start) * 1000
    write_times.append(write_time)


total_end = time.time()
total_elapsed = total_end - total_start

# --- RELEASE EVERYTHING ---
cap.release()
out.release() 
cv2.destroyAllWindows()

# --- PRINT METRICS ---
if inference_times:
    inference_times = inference_times[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(inference_times, label="Inference time (ms)", color='blue')
    plt.xlabel("Frame number")
    plt.ylabel("Inference time (ms)")
    plt.title("YOLO Inference Time Per Frame")
    plt.grid(True)
    plt.legend()
    plt.savefig("inference_time_plot.png")
    
    avg_read = sum(read_times) / len(read_times)
    avg_plot = sum(plot_times) / len(plot_times)
    avg_write = sum(write_times) / len(write_times)
    
    # Calculate Total Overhead
    total_overhead = avg_read + avg_plot + avg_write
    
    print("\n--- Detailed Performance Metrics (ms) ---")
    print(f"Average Inference (GPU): {sum(inference_times)/len(inference_times):.2f} ms") 
    print(f"Average Read/Capture (I/O): {avg_read:.2f} ms")
    print(f"Average Plot/Drawing (CPU): {avg_plot:.2f} ms")
    print(f"Average Write/Encoding (I/O/CPU): {avg_write:.2f} ms")
    print(f"Total Overhead: {total_overhead:.2f} ms")
    print(f"Theoretical Total TPF: {(sum(inference_times)/len(inference_times) + total_overhead):.2f} ms")

print(f"Total time elapsed: {(total_elapsed):.2f} s")
print(f"Total frames processed: {frames}")
print(f"Camera FPS (Source Video FPS): {(fps):.2f}")
print(f"Actual Processing FPS: {(frames / (total_elapsed)):.2f}")
print(f"\nSuccessfully saved annotated video to: {output_path}")