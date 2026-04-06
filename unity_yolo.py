from ultralytics import YOLO
import cv2
import numpy as np
import mss
import socket
import pyautogui

screen_width, screen_height = pyautogui.size()

model = YOLO("yolov8n.pt")

sct = mss.mss()

monitor = {
    "top": 100,
    "left": 110,
    "width": 900,
    "height": 1400
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
unity_address = ("127.0.0.1", 5055)

while True:

    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    results = model(frame)

    annotated = results[0].plot()

    frame_width = frame.shape[1]

    left_zone = False
    center_zone = False
    right_zone = False

    for box in results[0].boxes:

        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = box.xyxy[0]
        object_center = (x1 + x2) / 2

        if label in ["person","car","truck","bus"]:

            if object_center < frame_width/3:
                left_zone = True

            elif object_center < 2*frame_width/3:
                center_zone = True

            else:
                right_zone = True

    # Decision logic

    if center_zone:
        command = "STOP"

    elif left_zone:
        command = "RIGHT"

    elif right_zone:
        command = "LEFT"

    else:
        command = "FORWARD"
    print("Sent:", command)

    sock.sendto(command.encode(), unity_address)

    print("Sent:", command)

    cv2.imshow("YOLO Output", annotated)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()