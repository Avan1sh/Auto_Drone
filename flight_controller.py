"""
MAVSDK Flight Controller
=========================
All flight commands go through MAVSDK/PX4 instead of AirSim SimpleFlight.
Same code works on PX4 SITL (sim) and real PX4 hardware.
"""

import asyncio
from mavsdk import System
from mavsdk.offboard import (
    OffboardError,
    VelocityBodyYawspeed,
    VelocityNedYaw,
)
from mavsdk.action import ActionError


class FlightController:
    """Async MAVSDK flight controller for PX4 drones."""

    def __init__(self, system_address: str = "udp://:14540"):
        self.system_address = system_address
        self.drone = System()
        self.is_connected = False
        self.is_flying = False
        self.is_offboard = False

    async def connect(self):
        """Connect to PX4 and wait for GPS lock."""
        print(f"[FC] Connecting to {self.system_address}...")
        await self.drone.connect(system_address=self.system_address)

        print("[FC] Waiting for connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[FC] Connected!")
                break

        print("[FC] Waiting for position estimate...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("[FC] Position OK")
                break

        self.is_connected = True
        print("[FC] Ready\n")

    async def arm(self):
        """Arm motors."""
        print("[FC] Arming...")
        await self.drone.action.arm()
        print("[FC] Armed")

    async def disarm(self):
        """Disarm motors."""
        await self.drone.action.disarm()
        self.is_flying = False
        print("[FC] Disarmed")

    async def takeoff(self, altitude: float = 3.0):
        """Arm and takeoff to altitude (meters)."""
        await self.arm()
        await self.drone.action.set_takeoff_altitude(altitude)
        print(f"[FC] Taking off to {altitude}m...")
        await self.drone.action.takeoff()

        async for pos in self.drone.telemetry.position():
            if pos.relative_altitude_m >= altitude * 0.90:
                break

        self.is_flying = True
        print(f"[FC] Takeoff complete\n")

    async def land(self):
        """Land at current position."""
        print("[FC] Landing...")
        if self.is_offboard:
            await self.stop_offboard()
        await self.drone.action.land()

        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                break

        self.is_flying = False
        print("[FC] Landed")

    async def start_offboard(self):
        """Enter offboard mode for velocity control."""
        if self.is_offboard:
            return
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0, 0, 0, 0)
        )
        await self.drone.offboard.start()
        self.is_offboard = True
        print("[FC] Offboard started")

    async def stop_offboard(self):
        """Exit offboard mode."""
        if not self.is_offboard:
            return
        await self.drone.offboard.stop()
        self.is_offboard = False

    async def move_body(self, forward: float, right: float,
                        down: float, yaw_rate: float = 0.0):
        """Velocity in body frame. Replaces AirSim moveByVelocityAsync.

        Args:
            forward: m/s (+ forward)
            right:   m/s (+ right)
            down:    m/s (+ down, NED)
            yaw_rate: deg/s (+ clockwise)
        """
        if not self.is_offboard:
            await self.start_offboard()
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(forward, right, down, yaw_rate)
        )

    async def move_ned(self, north: float, east: float,
                       down: float, yaw: float = 0.0):
        """Velocity in NED (world) frame."""
        if not self.is_offboard:
            await self.start_offboard()
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(north, east, down, yaw)
        )

    async def hover(self):
        """Hold position."""
        await self.move_body(0, 0, 0, 0)

    async def get_altitude(self) -> float:
        """Current relative altitude in meters."""
        async for pos in self.drone.telemetry.position():
            return pos.relative_altitude_m

    async def get_position_ned(self) -> tuple:
        """Current position from PX4 telemetry in local NED frame.

        Returns:
            (x, y, z) where:
              x = North  (meters from home, + = forward/north)
              y = East   (meters from home, + = right/east)
              z = Up     (meters above home, + = up)

        Hardware-portable replacement for:
            client.getMultirotorState().kinematics_estimated.position
        """
        async for pv in self.drone.telemetry.position_velocity_ned():
            p = pv.position
            return p.north_m, p.east_m, -p.down_m   # negate Z: NED→up-positive

    async def get_velocity_ned(self) -> tuple:
        """Current velocity from PX4 telemetry in NED frame.

        Returns:
            (vx, vy, vz) where:
              vx = North m/s  (+ = forward)
              vy = East  m/s  (+ = right)
              vz = Up    m/s  (+ = up, negated from NED down)

        Hardware-portable replacement for:
            client.getMultirotorState().kinematics_estimated.linear_velocity
        """
        async for vel in self.drone.telemetry.velocity_ned():
            return vel.north_m_s, vel.east_m_s, -vel.down_m_s  # negate Z


    async def get_telemetry(self) -> dict:
        """Snapshot of position, velocity, battery."""
        t = {}
        async for pos in self.drone.telemetry.position():
            t["alt_rel"] = pos.relative_altitude_m
            t["lat"] = pos.latitude_deg
            t["lon"] = pos.longitude_deg
            break
        async for vel in self.drone.telemetry.velocity_ned():
            t["vel_n"] = vel.north_m_s
            t["vel_e"] = vel.east_m_s
            t["vel_d"] = vel.down_m_s
            break
        async for bat in self.drone.telemetry.battery():
            t["battery_pct"] = bat.remaining_percent
            break
        return t

    async def shutdown(self):
        """Land, disarm, disconnect."""
        if self.is_flying:
            await self.land()
        await self.disarm()
        print("[FC] Shutdown complete")
