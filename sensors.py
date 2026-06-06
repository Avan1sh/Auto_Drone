"""
AirSim Sensor Interface
========================
Keeps AirSim API usage strictly for sensor data (cameras, depth, lidar).
Flight control is handled separately by MAVSDK via flight_controller.py.

This separation means the same sensor code works whether the drone is
controlled by SimpleFlight, PX4, or eventually a real flight controller.
"""

import airsim
import numpy as np
import cv2
from typing import Optional, Tuple


class DepthSensor:
    """Reads depth frames from AirSim's DepthPlanar camera.

    Optionally injects Gaussian noise to improve sim-to-real transfer.
    """

    def __init__(self, client: airsim.MultirotorClient,
                 camera_name: str = "front_center",
                 noise_std: float = 0.02):
        """
        Args:
            client:      Active AirSim client connection.
            camera_name: AirSim camera ID (must match settings.json).
            noise_std:   Std-dev of Gaussian noise added to depth (meters).
                         Set to 0.0 to disable noise injection.
        """
        self.client = client
        self.camera_name = camera_name
        self.noise_std = noise_std

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Capture a single depth frame from AirSim.

        Returns:
            2D numpy array of depth values in meters, or None on failure.
            Noise is injected if noise_std > 0.
        """
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    self.camera_name,
                    airsim.ImageType.DepthPlanar,
                    pixels_as_float=True,
                    compress=False
                )
            ])

            if not responses or responses[0].width == 0:
                return None

            # Convert flat float array → 2D depth image
            depth = airsim.list_to_2d_float_array(
                responses[0].image_data_float,
                responses[0].width,
                responses[0].height
            )

            # Inject sensor noise for sim-to-real robustness
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std, depth.shape)
                depth = depth + noise

            return depth

        except Exception as e:
            print(f"[DepthSensor] Frame capture failed: {e}")
            return None

    def analyze_zones(self, depth: np.ndarray,
                      num_zones: int = 3) -> Tuple[float, ...]:
        """Split depth frame into vertical zones and compute mean distance.

        Args:
            depth:     2D depth array from get_depth_frame().
            num_zones: Number of vertical strips (default 3: left/center/right).

        Returns:
            Tuple of mean distances (meters) for each zone, left to right.
        """
        h, w = depth.shape
        zone_width = w // num_zones
        results = []

        for i in range(num_zones):
            start_col = i * zone_width
            end_col = (i + 1) * zone_width if i < num_zones - 1 else w
            zone = depth[:, start_col:end_col]
            clipped = np.clip(zone, 0.1, 100.0)
            results.append(float(np.mean(clipped)))

        return tuple(results)


class RGBCamera:
    """Reads RGB frames from AirSim's Scene camera."""

    def __init__(self, client: airsim.MultirotorClient,
                 camera_name: str = "front_center"):
        self.client = client
        self.camera_name = camera_name

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a single RGB frame from AirSim.

        Returns:
            BGR numpy array (OpenCV format), or None on failure.
        """
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    self.camera_name,
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=True
                )
            ])

            if not responses or len(responses[0].image_data_uint8) == 0:
                return None

            png = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img = cv2.imdecode(png, cv2.IMREAD_COLOR)
            return img

        except Exception as e:
            print(f"[RGBCamera] Frame capture failed: {e}")
            return None


class SensorSuite:
    """Bundles AirSim camera sensors into a single interface.

    Scope: cameras and image processing ONLY.
    Position, velocity, and altitude come from flight_controller.py
    via MAVSDK telemetry — that data is hardware-portable.
    AirSim's getMultirotorState() is NOT used here.

    Usage:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        sensors = SensorSuite(client)

        depth = sensors.depth.get_depth_frame()
        rgb   = sensors.rgb.get_frame()
        left, center, right = sensors.depth.analyze_zones(depth)
    """

    def __init__(self, client: airsim.MultirotorClient,
                 camera_name: str = "front_center",
                 depth_noise: float = 0.02):
        self.client = client
        self.depth = DepthSensor(client, camera_name, noise_std=depth_noise)
        self.rgb = RGBCamera(client, camera_name)
