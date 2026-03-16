from ultralytics import YOLO
import sys

# 1. Load model NCNN (đã tối ưu cho CPU Pi 5)
model = YOLO('/home/pi/work/model/last26_ncnn_model', task='detect')

# 2. Chạy Tracking với cấu hình tối ưu nhất
# Sử dụng stream=True để không bị tràn RAM
results = model.track(
    source='/home/pi/work/data/testvideo1.mp4',
    conf=0.3,               # Ngưỡng thấp một chút để ByteTrack làm việc tốt hơn
    iou=0.5, 
    imgsz=480,              # Giữ nguyên 320 để duy trì tốc độ
    persist=True,           # BẮT BUỘC để ByteTrack hoạt động
    tracker="bytetrack.yaml", # Sử dụng ByteTrack (nhẹ hơn BoT-SORT)
    save=True,              # Lưu kết quả vào folder results
    project='/home/pi/work/results',
    name='tracking_test',
    exist_ok=True,
    vid_stride=2,           # AI chạy 1 frame, nghỉ 1 frame (Tăng FPS cực mạnh)
    classes=[0],            # Chỉ track người (Lecturer)
    stream=True             # Chạy dạng generator để tiết kiệm tài nguyên
)

print("\n--- Tracking ... (ByteTrack) ---")

for r in results:
    # Tính toán FPS dựa trên tốc độ xử lý của Model
    total_ms = sum(r.speed.values())
    fps = 1000 / total_ms if total_ms > 0 else 0
    
    # In ra Terminal để Bảo theo dõi
    sys.stdout.write(f"\r>>> Tracking FPS: {fps:.2f} | Object Count: {len(r.boxes)}   ")
    sys.stdout.flush()

print("\n --- Done! Video saved to work/results/tracking_test ---")
