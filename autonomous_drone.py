import airsim
import numpy as np
import cv2
import time

# ── Config ────────────────────────────────────────────────
SPEED         = 6.0
AVOID_SPEED   = 2.5
DANGER_DIST   = 25.0
CAUTION_DIST  = 38.0
SMOOTH        = 0.7
MIN_FRAMES    = 8
TARGET_ALT    = -3.0
ALT_KP        = 0.8
MAX_VZ        = 1.5
NAV_SPEED     = 1.0
# ── Replace single destination with waypoint list ─────────
WAYPOINTS   = [
( 20.0,   0.0),   # 50m straight ahead
    ( 50.0,  30.0),   # turn right 30m
    (  0.0,  30.0),   # fly back left
    (  0.0,   0.0),   # return to start
]
ARRIVE_DIST = 2.0

# ── State ─────────────────────────────────────────────────
steering_state = "FORWARD"
steering_count = 0
smooth_left    = 50.0
smooth_center  = 50.0
smooth_right   = 50.0

# ── Waypoint state ────────────────────────────────────────
current_waypoint = 0   # index into WAYPOINTS list

# ── Connect ───────────────────────────────────────────────
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# ── Functions ─────────────────────────────────────────────
def get_depth_frame():
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
    ])
    return airsim.get_pfm_array(responses[0])

def get_rgb_frame():
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
    ])
    png = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    return cv2.imdecode(png, cv2.IMREAD_COLOR)

def analyze_depth(depth):
    h, w  = depth.shape
    third = w // 3
    def safe_mean(zone):
        clipped = np.clip(zone, 0.1, 100.0)
        return float(np.mean(clipped))
    return (safe_mean(depth[:, :third]),
            safe_mean(depth[:, third:2*third]),
            safe_mean(depth[:, 2*third:]))

def get_altitude_correction():
    pos     = client.getMultirotorState().kinematics_estimated.position
    current = pos.z_val
    error   = TARGET_ALT - current
    vz      = max(-MAX_VZ, min(MAX_VZ, ALT_KP * error))
    if abs(error) > 0.5:
        print(f"Alt — current: {current:.1f}m  target: {TARGET_ALT}m  vz: {vz:.2f}")
    return vz

def get_navigation_velocity(current_x, current_y):
    global current_waypoint

    # All waypoints done
    if current_waypoint >= len(WAYPOINTS):
        return 0, 0, 0

    dest_x, dest_y = WAYPOINTS[current_waypoint]
    dx       = dest_x - current_x
    dy       = dest_y - current_y
    distance = np.sqrt(dx**2 + dy**2)

    # Reached this waypoint — advance to next
    if distance < ARRIVE_DIST:
        print(f"\n✔ Waypoint {current_waypoint + 1}/{len(WAYPOINTS)} reached!")
        current_waypoint += 1
        if current_waypoint >= len(WAYPOINTS):
            print("All waypoints complete!")
            return 0, 0, 0
        print(f"→ Heading to waypoint {current_waypoint + 1}: {WAYPOINTS[current_waypoint]}")
        dest_x, dest_y = WAYPOINTS[current_waypoint]
        dx       = dest_x - current_x
        dy       = dest_y - current_y
        distance = np.sqrt(dx**2 + dy**2)

    vx = (dx / distance) * NAV_SPEED
    vy = (dy / distance) * NAV_SPEED
    return vx, vy, distance

def decide(left, center, right):
    global steering_state, steering_count
    vz = get_altitude_correction()
    steering_count += 1

    pos                      = client.getMultirotorState().kinematics_estimated.position
    nav_vx, nav_vy, distance = get_navigation_velocity(pos.x_val, pos.y_val)

    # ── All waypoints done ────────────────────────────────
    if current_waypoint >= len(WAYPOINTS):
        print("Mission complete — hovering")
        return 0, 0, vz

    print(f"WP{current_waypoint + 1}/{len(WAYPOINTS)} — {distance:.1f}m away", end="  ")

    # ── All blocked — climb ───────────────────────────────
    if left < 5.0 and center < 5.0 and right < 5.0:
        steering_state = "UP"
        steering_count = 0
        print("EMERGENCY — climbing")
        return 0, 0, -2.0

    # ── Clear — navigate ──────────────────────────────────
    if center > CAUTION_DIST and left > CAUTION_DIST * 0.6 and right > CAUTION_DIST * 0.6:
        steering_state = "FORWARD"
        steering_count = 0
        print("CLEAR — navigating")
        return nav_vx, nav_vy, vz

    # ── Danger ────────────────────────────────────────────
    if center < DANGER_DIST:
        if steering_count >= MIN_FRAMES or steering_state == "FORWARD":
            steering_state = "LEFT" if left > right else "RIGHT"
            steering_count = 0
        if steering_state == "LEFT":
            print("DANGER — avoiding LEFT")
            return 0, -AVOID_SPEED, vz
        else:
            print("DANGER — avoiding RIGHT")
            return 0, AVOID_SPEED, vz

    # ── Caution ───────────────────────────────────────────
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

def draw_hud(rgb, left, center, right):
    h, w = rgb.shape[:2]
    t    = w // 3
    def bar_color(d):
        if d > CAUTION_DIST: return (0, 200, 0)
        if d > DANGER_DIST:  return (0, 165, 255)
        return (0, 0, 255)
    cv2.line(rgb, (t, 0),   (t, h),   (200, 200, 200), 1)
    cv2.line(rgb, (2*t, 0), (2*t, h), (200, 200, 200), 1)
    for val, x in [(left, t//2), (center, t + t//2), (right, 2*t + t//2)]:
        color = bar_color(val)
        cv2.putText(rgb, f"{val:.1f}m", (x-18, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        bar_h = int(np.clip((1 - val/30) * 40, 2, 40))
        cv2.rectangle(rgb, (x-15, h-bar_h), (x+15, h), color, -1)

    # Draw waypoint progress on HUD
    pos = client.getMultirotorState().kinematics_estimated.position
    if current_waypoint < len(WAYPOINTS):
        _, _, distance = get_navigation_velocity(pos.x_val, pos.y_val)
        wp_text = f"WP {current_waypoint + 1}/{len(WAYPOINTS)} — {distance:.1f}m"
    else:
        wp_text = "Mission complete!"
    cv2.putText(rgb, wp_text, (4, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return rgb

# ── Takeoff ───────────────────────────────────────────────
print("Taking off...")
client.takeoffAsync().join()

print("Climbing to target altitude...")
client.moveToZAsync(TARGET_ALT, 2).join()
time.sleep(2)

pos = client.getMultirotorState().kinematics_estimated.position
print(f"Position — X: {pos.x_val:.1f}  Y: {pos.y_val:.1f}  Z: {pos.z_val:.1f}")
print(f"Waypoints: {WAYPOINTS}")
print(f"→ Heading to waypoint 1: {WAYPOINTS[0]}")
print("Autonomous navigation running — press Q to quit")

# ── Main loop ─────────────────────────────────────────────
try:
    while True:
        depth = get_depth_frame()
        rgb   = get_rgb_frame()

        if depth is None or rgb is None or depth.size == 0 or rgb.size == 0:
            print("Warning — empty frame, skipping")
            continue

        raw_left, raw_center, raw_right = analyze_depth(depth)

        smooth_left   = SMOOTH * smooth_left   + (1 - SMOOTH) * raw_left
        smooth_center = SMOOTH * smooth_center + (1 - SMOOTH) * raw_center
        smooth_right  = SMOOTH * smooth_right  + (1 - SMOOTH) * raw_right

        vx, vy, vz = decide(smooth_left, smooth_center, smooth_right)

        # Check mission complete
        if current_waypoint >= len(WAYPOINTS):
            print("All waypoints reached — landing!")
            break

        client.moveByVelocityAsync(vx, vy, vz, 0.5)

        rgb = draw_hud(rgb, smooth_left, smooth_center, smooth_right)
        cv2.imshow("Drone — depth avoidance + navigation", rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Landing...")
    cv2.destroyAllWindows()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Done.")