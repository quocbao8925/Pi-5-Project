from ultralytics import YOLO
import os

# 1. Load model
model = YOLO('/home/pi/work/model/last26_ncnn_model', task='detect')

# 2. Chạy và ép lưu vào một chỗ cố định
# Lưu ý: Mình bỏ format='mp4' và dùng imgsz=320 cho nhẹ Pi 5
model.predict(source='/home/pi/work/data/testvideo.mp4',
              save=True,
              project='/home/pi/work/results', # Lưu thẳng vào đây
              name='latest_run',               # Thư mục con cố định
              imgsz=480,
              conf=0.5,
              exist_ok=True)                   # Ghi đè lên folder cũ nếu có

print("Checking...")
os.system("ls -R /home/pi/work/results/latest_run")
