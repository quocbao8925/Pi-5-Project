import cv2
import time
from ultralytics import YOLO

model = YOLO('/home/pi/work/model/last26_ncnn_model', task='detect')
cap = cv2.VideoCapture('/home/pi/work/data/testvideo.mp4')

# Lấy thông số để lưu video
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter('/home/pi/work/results/fast_output.mp4', 
                     cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

prev_time = 0
for r in model.predict(source='/home/pi/work/data/testvideo.mp4', stream=True, imgsz=320):
    frame = r.orig_img # Lấy ảnh gốc (chưa vẽ gì) cho nhẹ
    
    # 1. Tự vẽ box thủ công (Nhẹ hơn r.plot() rất nhiều)
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 2. Tính và vẽ FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 1, 1.5, (0, 0, 255), 2)

    out.write(frame)
    print(f"\rInference FPS: {fps:.1f}", end="")

cap.release()
out.release()