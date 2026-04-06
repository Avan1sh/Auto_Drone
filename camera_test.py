import airsim
import numpy as np
import cv2

client = airsim.MultirotorClient()
client.confirmConnection()

responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
])

response = responses[0]

png_bytes = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
img = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imwrite("drone_view.png", img)
print("Saved! Open drone_view.png — should show the sim world now")