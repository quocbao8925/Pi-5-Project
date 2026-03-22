import cv2
import time
import argparse
import threading
import os
from queue import Queue
from ultralytics import YOLO

def main():
    # 1. Argument Parser Configuration
    parser = argparse.ArgumentParser(description="YOLO Granular Benchmarking for RPi 5")
    parser.add_argument('--model', type=str, default='/home/pi/work/model/yolo261.pt')
    parser.add_argument('--source', type=str, default='/home/pi/work/data/xuccat.mp4')
    parser.add_argument('--img', type=int, default=320)
    # Customize interval of log
    parser.add_argument('--interval', type=int, default=3, help='Log every N frames')
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.expanduser(args.model))
    source_path = os.path.abspath(os.path.expanduser(args.source))

    # 2. Model & Queue Initialization
    model = YOLO(model_path, task='detect')
    frame_queue = Queue(maxsize=30)
    stop_event = threading.Event()

    # 3. Engine Warm-up
    cap_temp = cv2.VideoCapture(source_path)
    ret, warm_frame = cap_temp.read()
    cap_temp.release()
    if not ret: return

    print(f"[*] Warming up...")
    for _ in range(5):
        model.predict(source=warm_frame, imgsz=args.img, verbose=False)

    # 4. Video Reader Thread
    def video_reader(path, queue, stop_sig):
        cap = cv2.VideoCapture(path)
        while not stop_sig.is_set():
            ret, frame = cap.read()
            if not ret: break
            queue.put(frame)
        cap.release()
        stop_sig.set()

    # 5. Main Benchmark Loop
    print(f"[*] Benchmark started (Logging every {args.interval} frames)...")
    reader_thread = threading.Thread(target=video_reader, args=(source_path, frame_queue, stop_event))
    reader_thread.start()

    frame_count = 0
    start_time = time.time()
    last_time = start_time

    try:
        while not stop_event.is_set() or not frame_queue.empty():
            if not frame_queue.empty():
                frame = frame_queue.get()
                
                # Inference
                model.predict(source=frame, imgsz=args.img, conf=0.5, max_det=10, verbose=False)
                frame_count += 1
                
                # Granular Logging
                if frame_count % args.interval == 0:
                    curr_time = time.time()
                    # Tính FPS cho khoảng interval vừa qua
                    interval_duration = curr_time - last_time
                    if interval_duration > 0:
                        interval_fps = args.interval / interval_duration
                        # Dùng \r để log cập nhật trên cùng một dòng nếu muốn gọn, 
                        # hoặc print bình thường để lưu lịch sử. Ở đây mình dùng print nhé.
                        print(f"[-] Frame: {frame_count:5} | Segment FPS: {interval_fps:5.2f}")
                    last_time = curr_time
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        stop_event.set()

    # 6. Final Summary
    total_time = time.time() - start_time
    print("\n" + "="*45)
    print(f"  FINAL AVERAGE FPS: {frame_count / total_time:.2f}")
    print("="*45)

    stop_event.set()
    reader_thread.join()

if __name__ == "__main__":
    main()
