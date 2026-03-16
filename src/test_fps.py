import cv2
import time
from ultralytics import YOLO

# Configuration
MODEL_PATH = '/home/pi/work/model/last26_ncnn_model'
VIDEO_PATH = '/home/pi/work/data/testvideo1.mp4'
IMG_SIZE = 480

# 1. Initialize model and video source
# Using task='detect' explicitly for NCNN
model = YOLO(MODEL_PATH, task='detect')
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# 2. Warm-up phase
# Run a few frames to initialize CPU cache and NCNN buffers
print("Warming up engine...")
for _ in range(10):
    ret, frame = cap.read()
    if not ret: break
    model.predict(frame, imgsz=IMG_SIZE, verbose=False)

# 3. Peak Performance Benchmark
print(f"Starting Benchmark at imgsz={IMG_SIZE}...")
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # End of video
    
    # Run inference only (No plotting, No saving, No logging)
    model.predict(frame, imgsz=IMG_SIZE, verbose=False)
    frame_count += 1

    # Optional: Periodic progress check every 100 frames
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

end_time = time.time()

# 4. Final Calculations
total_time = end_time - start_time
avg_fps = frame_count / total_time

print("\n" + "="*40)
print(f"PEAK PERFORMANCE RESULTS")
print(f"Total Frames: {frame_count}")
print(f"Total Time  : {total_time:.2f} seconds")
print(f"Average FPS : {avg_fps:.2f}")
print("="*40)

cap.release()
