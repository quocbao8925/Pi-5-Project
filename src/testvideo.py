from ultralytics import YOLO
import os
import argparse

# 1. Load model
parser = argparse.ArgumentParser(description="YOLO testvideo")
parser.add_argument('--model', type=str, default='/home/pi/work/model/last26_480_ncnn_model')
parser.add_argument('--source', type=str, default='/home/pi/work/data/xuccat.mp4')
parser.add_argument('--img', type=int, default=480)
args = parser.parse_args()
model_path = os.path.abspath(os.path.expanduser(args.model))
source_path = os.path.abspath(os.path.expanduser(args.source))
model = YOLO(model_path, task='detect')
# 2. Chạy và ép lưu vào một chỗ cố định
# Lưu ý: Mình bỏ format='mp4' và dùng imgsz=320 cho nhẹ Pi 5
model.predict(source=source_path,
              save=True,
              project='/home/pi/work/results', # Lưu thẳng vào đây
              name='latest_run',               # Thư mục con cố định
              imgsz=320,
              conf=0.5,
              exist_ok=True)                   # Ghi đè lên folder cũ nếu có

print("Checking...")
os.system("ls -R /home/pi/work/results/latest_run")
