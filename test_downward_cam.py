import airsim
client = airsim.MultirotorClient()
response = client.simGetImages([
    airsim.ImageRequest("bottom_center", airsim.ImageType.Scene)
])
print(len(response[0].image_data_uint8))  # Should be non-zero