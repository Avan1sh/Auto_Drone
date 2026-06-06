# PX4 SITL Setup Guide (WSL2 + AirSim)

## Architecture

```
┌─────────────────────────┐      UDP       ┌──────────────────────┐
│   Windows               │  ◄──────────►  │   WSL2 / Linux       │
│                         │   port 4560    │                      │
│  AirSim (UE4)           │                │  PX4 SITL Autopilot  │
│  + Python scripts       │   port 14540   │  (none_iris)         │
│  + MAVSDK               │  ◄──────────►  │                      │
└─────────────────────────┘                └──────────────────────┘
```

## Prerequisites

- Windows 10/11 with WSL2 enabled
- Ubuntu 20.04+ on WSL2
- Unreal Engine 4.27 + AirSim plugin (already set up)

---

## Step 1 — Install WSL2 (if not already)

```powershell
# In PowerShell (Admin)
wsl --install -d Ubuntu-22.04
```

## Step 2 — Clone and build PX4 in WSL2

```bash
# In WSL2 terminal
cd ~

# Clone PX4
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot

# Run the setup script (installs all dependencies)
bash ./Tools/setup/ubuntu.sh

# Build SITL (no GUI, for AirSim integration)
make px4_sitl_default none_iris
```

> **Note:** First build takes 10-15 minutes. Subsequent builds are fast.

## Step 3 — Configure AirSim for PX4

The `settings.json` has already been updated at:
```
C:\Users\avani\Documents\AirSim\settings.json
```

Key settings:
- `VehicleType`: `PX4Multirotor` (was `SimpleFlight`)
- `UseTcp`: `true`, `TcpPort`: `4560`
- `LockStep`: `true` (deterministic simulation)

## Step 4 — Install MAVSDK Python package

```powershell
# In Windows PowerShell, inside your venv
cd C:\Users\avani\drone_ai\Auto_Drone
.\Scripts\activate
pip install mavsdk
```

## Step 5 — Find WSL2 IP address

```bash
# In WSL2
hostname -I
# Example output: 172.20.10.5
```

Update `settings.json` → `LocalHostIp` to this IP if connection fails.

## Step 6 — Run everything

### Terminal 1: Start PX4 SITL (WSL2)
```bash
cd ~/PX4-Autopilot
make px4_sitl_default none_iris
```

Wait for: `INFO  [commander] Ready for takeoff!`

### Terminal 2: Start AirSim (Windows)
Open Unreal Engine → AUTO_DRONE project → Press Play

### Terminal 3: Run the drone (Windows)
```powershell
cd C:\Users\avani\drone_ai\Auto_Drone
.\Scripts\activate
python autonomous_drone_px4.py
```

---

## Troubleshooting

### PX4 can't connect to AirSim
- Check Windows Firewall allows UDP ports 4560, 14540, 14580
- Verify WSL2 IP in `settings.json` → `LocalHostIp`

### "Offboard rejected"
- PX4 requires a stream of setpoints before entering offboard mode
- The `FlightController` handles this automatically (sends hover setpoint first)

### Altitude drift
- The P-controller in the drone script compensates; tune `ALT_KP` if needed
- PX4's internal altitude controller also runs — they work together

### Slow simulation
- `LockStep: true` means PX4 waits for AirSim — sim runs at physics speed
- If too slow, try reducing camera resolution in `settings.json`

---

## File Map After Migration

```
Auto_Drone/
├── autonomous_drone.py        ← OLD (SimpleFlight, still works)
├── autonomous_drone_px4.py    ← NEW (MAVSDK + PX4)
├── flight_controller.py       ← NEW (MAVSDK wrapper)
├── sensors.py                 ← NEW (AirSim sensors only)
├── requirements.txt           ← Updated with mavsdk
├── yolo_detection.py          ← Unchanged
├── unity_yolo.py              ← Unchanged
├── camera_test.py             ← Unchanged
└── yolov8n.pt                 ← Unchanged
```
