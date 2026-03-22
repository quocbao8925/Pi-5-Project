import cv2
import time
from ultralytics import YOLO

# Khai báo đường dẫn tuyệt đối
model = YOLO('/home/pi/work/model/last26h_ncnn_model') 
VIDEO_PATH = '/home/pi/work/data/testvideo_gmnrcm.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

fps_data = [] # Mảng chứa text để ghi file
frame_count = 0

print("🚀 Bắt đầu ép xung CPU để quét video và ghi log...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    start_time = time.time()
    # Chạy YOLO tinh gọn nhất có thể
    results = model(frame, imgsz=320, conf=0.5, verbose=False)
    inference_time = time.time() - start_time
    
    if inference_time > 0:
        fps = 1.0 / inference_time
        frame_count += 1
        # Ghi nhận dữ liệu theo chuẩn: Frame,FPS
        fps_data.append(f"{frame_count},{fps:.2f}\n")
        
        # In nhẹ ra terminal mỗi 10 frame cho đỡ lag màn hình
        if frame_count % 10 == 0:
            print(f"Đã xử lý {frame_count} frames...")

cap.release()

# Xuất dữ liệu ra file text siêu nhẹ
log_file = '/home/pi/work/src/fps_log.txt'
with open(log_file, 'w') as f:
    f.writelines(fps_data)

print(f"✅ Xong! Đã xuất dữ liệu thô ra file {log_file}")
