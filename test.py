# Run this separately to find your starting position
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
pos = client.getMultirotorState().kinematics_estimated.position
print(f"X: {pos.x_val:.1f}  Y: {pos.y_val:.1f}  Z: {pos.z_val:.1f}")



