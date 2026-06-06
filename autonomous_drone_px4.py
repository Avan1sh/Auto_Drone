"""
Autonomous Drone — PX4 + AirSim Sensors
=========================================
Migrated from pure AirSim SimpleFlight to:
  - MAVSDK/PX4 for flight control (hardware-portable)
  - AirSim for sensor data only (depth camera, RGB camera)

Run PX4 SITL first (in WSL2):
    cd PX4-Autopilot && make px4_sitl_default none_iris

Then run this script (on Windows):
    python autonomous_drone_px4.py
"""

import asyncio
import airsim
import numpy as np
import cv2
import time

from sensors import SensorSuite
from flight_controller import FlightController

# ── Config ────────────────────────────────────────────────
SPEED         = 6.0
AVOID_SPEED   = 2.5
DANGER_DIST   = 25.0
CAUTION_DIST  = 38.0
SMOOTH        = 0.7
MIN_FRAMES    = 8
TARGET_ALT    = 3.0      # meters above ground (positive = up)
ALT_KP        = 0.8
MAX_VZ        = 1.5
NAV_SPEED     = 1.0

# ── Waypoints ─────────────────────────────────────────────
WAYPOINTS = [
    (20.0,   0.0),    # straight ahead
    (50.0,  30.0),    # turn right
    ( 0.0,  30.0),    # fly back left
    ( 0.0,   0.0),    # return to start
]
ARRIVE_DIST = 2.0


class AutonomousDrone:
    """Autonomous navigation with depth-based obstacle avoidance.

    Flight control: MAVSDK → PX4
    Sensors:        AirSim (depth + RGB cameras)
    """

    def __init__(self):
        # Flight controller (MAVSDK/PX4)
        self.fc = FlightController(system_address="udp://:14540")

        # AirSim client (sensors only)
        self.airsim_client = airsim.MultirotorClient()

        # Sensor suite (wraps AirSim camera/state APIs)
        self.sensors = SensorSuite(
            self.airsim_client,
            camera_name="front_center",
            depth_noise=0.02  # sim-to-real noise injection
        )

        # Navigation state
        self.current_waypoint = 0
        self.steering_state = "FORWARD"
        self.steering_count = 0

        # Smoothed depth readings
        self.smooth_left   = 50.0
        self.smooth_center = 50.0
        self.smooth_right  = 50.0

    async def initialize(self):
        """Connect to both PX4 and AirSim."""
        # Connect AirSim (sensors)
        print("=" * 60)
        print("  Autonomous Drone — PX4 + AirSim Sensors")
        print("=" * 60)
        print("\n[INIT] Connecting AirSim (sensors)...")
        self.airsim_client.confirmConnection()
        print("[INIT] AirSim connected\n")

        # Connect PX4 (flight control)
        await self.fc.connect()

    async def takeoff(self):
        """Takeoff to target altitude."""
        await self.fc.takeoff(altitude=TARGET_ALT)
        # Small settle time
        await asyncio.sleep(2)

        x, y, z = await self.fc.get_position_ned()   # PX4 telemetry, not AirSim
        print(f"Position — X: {x:.1f}  Y: {y:.1f}  Z: {z:.1f}")
        print(f"Waypoints: {WAYPOINTS}")
        print(f"→ Heading to waypoint 1: {WAYPOINTS[0]}")
        print("Autonomous navigation running — press Q to quit\n")

    def get_navigation_velocity(self, current_x, current_y):
        """Compute velocity vector toward current waypoint."""
        if self.current_waypoint >= len(WAYPOINTS):
            return 0, 0, 0

        dest_x, dest_y = WAYPOINTS[self.current_waypoint]
        dx = dest_x - current_x
        dy = dest_y - current_y
        distance = np.sqrt(dx**2 + dy**2)

        # Reached this waypoint
        if distance < ARRIVE_DIST:
            print(f"\n✔ Waypoint {self.current_waypoint + 1}/{len(WAYPOINTS)} reached!")
            self.current_waypoint += 1
            if self.current_waypoint >= len(WAYPOINTS):
                print("All waypoints complete!")
                return 0, 0, 0
            print(f"→ Heading to WP {self.current_waypoint + 1}: "
                  f"{WAYPOINTS[self.current_waypoint]}")
            dest_x, dest_y = WAYPOINTS[self.current_waypoint]
            dx = dest_x - current_x
            dy = dest_y - current_y
            distance = np.sqrt(dx**2 + dy**2)

        vx = (dx / distance) * NAV_SPEED
        vy = (dy / distance) * NAV_SPEED
        return vx, vy, distance

    async def get_altitude_correction(self):
        """P-controller for altitude hold."""
        alt = await self.fc.get_altitude()
        error = TARGET_ALT - alt
        vz = max(-MAX_VZ, min(MAX_VZ, ALT_KP * error))
        # NED: positive down, so negate for "go up"
        vz_ned = -vz
        if abs(error) > 0.5:
            print(f"Alt — current: {alt:.1f}m  target: {TARGET_ALT}m  vz: {vz_ned:.2f}")
        return vz_ned

    async def decide(self, left, center, right, drone_pos=None):
        """Obstacle avoidance + waypoint navigation decision engine."""
        vz = await self.get_altitude_correction()
        self.steering_count += 1

        # Use position passed in (already fetched from PX4 in main loop)
        # Falls back to fc telemetry if called standalone
        if drone_pos is None:
            drone_pos = await self.fc.get_position_ned()
        x, y, _ = drone_pos
        result = self.get_navigation_velocity(x, y)
        nav_vx, nav_vy = result[0], result[1]
        distance = result[2] if len(result) == 3 else 0

        # All waypoints done
        if self.current_waypoint >= len(WAYPOINTS):
            print("Mission complete — hovering")
            return 0, 0, vz

        print(f"WP{self.current_waypoint + 1}/{len(WAYPOINTS)} "
              f"— {distance:.1f}m away", end="  ")

        # Emergency — all blocked
        if left < 5.0 and center < 5.0 and right < 5.0:
            self.steering_state = "UP"
            self.steering_count = 0
            print("EMERGENCY — climbing")
            return 0, 0, -2.0  # NED: negative = up

        # Clear — navigate toward waypoint
        if (center > CAUTION_DIST and
                left > CAUTION_DIST * 0.6 and
                right > CAUTION_DIST * 0.6):
            self.steering_state = "FORWARD"
            self.steering_count = 0
            print("CLEAR — navigating")
            return nav_vx, nav_vy, vz

        # Danger — dodge
        if center < DANGER_DIST:
            if (self.steering_count >= MIN_FRAMES or
                    self.steering_state == "FORWARD"):
                self.steering_state = "LEFT" if left > right else "RIGHT"
                self.steering_count = 0
            if self.steering_state == "LEFT":
                print("DANGER — avoiding LEFT")
                return 0, -AVOID_SPEED, vz
            else:
                print("DANGER — avoiding RIGHT")
                return 0, AVOID_SPEED, vz

        # Caution — slow approach
        diff = left - right
        if abs(diff) < 8.0:
            print("Caution — slow nav")
            return nav_vx * 0.5, nav_vy * 0.5, vz
        elif diff > 0:
            print("Caution — drifting LEFT")
            return nav_vx * 0.4, -AVOID_SPEED * 0.3, vz
        else:
            print("Caution — drifting RIGHT")
            return nav_vx * 0.4, AVOID_SPEED * 0.3, vz

    def draw_hud(self, rgb, left, center, right, drone_pos=None):
        """Overlay depth + waypoint info on RGB frame."""
        h, w = rgb.shape[:2]
        t = w // 3

        def bar_color(d):
            if d > CAUTION_DIST:
                return (0, 200, 0)
            if d > DANGER_DIST:
                return (0, 165, 255)
            return (0, 0, 255)

        cv2.line(rgb, (t, 0), (t, h), (200, 200, 200), 1)
        cv2.line(rgb, (2*t, 0), (2*t, h), (200, 200, 200), 1)

        for val, x in [(left, t//2), (center, t + t//2),
                       (right, 2*t + t//2)]:
            color = bar_color(val)
            cv2.putText(rgb, f"{val:.1f}m", (x-18, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            bar_h = int(np.clip((1 - val/30) * 40, 2, 40))
            cv2.rectangle(rgb, (x-15, h-bar_h), (x+15, h), color, -1)

        # Waypoint progress — pos passed in from caller (already fetched via PX4)
        if self.current_waypoint < len(WAYPOINTS) and drone_pos is not None:
            _, _, dist = self.get_navigation_velocity(drone_pos[0], drone_pos[1])
            wp_text = (f"WP {self.current_waypoint + 1}/"
                       f"{len(WAYPOINTS)} — {dist:.1f}m")
        else:
            wp_text = "Mission complete!"

        # PX4 badge
        cv2.putText(rgb, "PX4+MAVSDK", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)
        cv2.putText(rgb, wp_text, (4, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return rgb

    async def run(self):
        """Main navigation loop."""
        await self.initialize()
        await self.takeoff()

        try:
            while True:
                # 1. Read sensors (AirSim only)
                depth = self.sensors.depth.get_depth_frame()
                rgb = self.sensors.rgb.get_frame()

                if depth is None or rgb is None:
                    print("Warning — empty frame, skipping")
                    await asyncio.sleep(0.05)
                    continue

                # 2. Analyze depth zones
                raw_left, raw_center, raw_right = \
                    self.sensors.depth.analyze_zones(depth)

                # 3. Smooth readings
                self.smooth_left = (SMOOTH * self.smooth_left +
                                    (1 - SMOOTH) * raw_left)
                self.smooth_center = (SMOOTH * self.smooth_center +
                                      (1 - SMOOTH) * raw_center)
                self.smooth_right = (SMOOTH * self.smooth_right +
                                     (1 - SMOOTH) * raw_right)

                # 4. Fetch drone position once (PX4 telemetry — hardware portable)
                drone_pos = await self.fc.get_position_ned()   # (x, y, z)

                # 5. Decide action
                vx, vy, vz = await self.decide(
                    self.smooth_left, self.smooth_center, self.smooth_right,
                    drone_pos=drone_pos
                )

                # 6. Check mission complete
                if self.current_waypoint >= len(WAYPOINTS):
                    print("All waypoints reached — landing!")
                    break

                # 7. Send velocity to PX4 via MAVSDK
                await self.fc.move_body(
                    forward=vx, right=vy, down=vz
                )

                # 8. Display HUD (reuse position already fetched — no second call)
                rgb = self.draw_hud(
                    rgb, self.smooth_left,
                    self.smooth_center, self.smooth_right,
                    drone_pos=drone_pos
                )
                cv2.imshow("Drone — PX4 + Depth Avoidance", rgb)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

                # Yield to event loop
                await asyncio.sleep(0.02)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print("Shutting down...")
            cv2.destroyAllWindows()
            await self.fc.shutdown()
            print("Done.")


async def main():
    drone = AutonomousDrone()
    await drone.run()


if __name__ == "__main__":
    asyncio.run(main())
