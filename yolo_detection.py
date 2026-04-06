import airsim
import numpy as np
import cv2
from ultralytics import YOLO

# Load your YOLO model
model = YOLO("yolov8n.pt")  # nano = fastest, good for real-time

# Connect to sim
client = airsim.MultirotorClient()
client.confirmConnection()

print("Running — press Q to quit")

while True:
    # 1. Grab frame from drone camera
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
    ])
    response = responses[0]

    # 2. Decode
    png_bytes = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)

    # 3. Run YOLO detection on the frame
    results = model(img, verbose=False)

    # 4. Draw detections onto the frame
    annotated = results[0].plot()

    # 5. Show live window
    cv2.imshow("Drone — YOLO vision", annotated)

    # 6. Print what was detected this frame
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        print(f"Detected: {label} ({conf:.0%} confidence)")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()