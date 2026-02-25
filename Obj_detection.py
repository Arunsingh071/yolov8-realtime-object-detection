import cv2
import time
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# -------------------------------
# Configuration
# -------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 800
FRAME_HEIGHT = 500
CONFIDENCE_THRESHOLD = 0.4

# -------------------------------
# Load YOLOv8 Model
# -------------------------------
model = YOLO("yolov8s.pt")  

# -------------------------------
# Initialize Camera
# -------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# -------------------------------
# GUI Setup
# -------------------------------
root = tk.Tk()
root.title("Real-Time Object Detection using YOLO")
root.geometry("900x650")

video_label = tk.Label(root)
video_label.pack()

# -------------------------------
# FPS Calculation
# -------------------------------
prev_time = 0

def run_detection():
    global prev_time

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # YOLO inference
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    # Convert to Tkinter format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, run_detection)

# -------------------------------
# Start Application
# -------------------------------
run_detection()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
