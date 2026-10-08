import mujoco
import mujoco.viewer
import numpy as np
import time

# ============================================================
# HEAL 6-DOF MODEL
# ============================================================

MODEL_PATH = "6DOF_HEAL_Manipulator.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# GET ACTUATOR IDs
# ============================================================

motor_names = [
    "motor1",
    "motor2",
    "motor3",
    "motor4",
    "motor5",
    "motor6"
]

motor_ids = []

for name in motor_names:
    motor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )
    motor_ids.append(motor_id)


# ============================================================
# INITIAL POSITION
# ============================================================

start_deg = np.array([
    0,
    0,
    0,
    0,
    0,
    0
], dtype=float)


# ============================================================
# TARGET POSITION
# ============================================================

target_deg = np.array([
    0,
    -90,
    90,
     0,
    0,
     0
], dtype=float)


start_rad = np.deg2rad(start_deg)
target_rad = np.deg2rad(target_deg)


# ============================================================
# SET INITIAL ROBOT POSITION
# ============================================================

data.qpos[:6] = start_rad
data.qvel[:6] = 0

for i in range(6):
    data.ctrl[motor_ids[i]] = start_rad[i]

mujoco.mj_forward(model, data)


# ============================================================
# SMOOTH INTERPOLATION
# ============================================================

def smooth_motion(t):

    # t must be between 0 and 1

    return 0.5 * (1 - np.cos(np.pi * t))


# ============================================================
# MOTION TIME
# ============================================================

motion_time = 5.0     # seconds
hold_time = 2.0       # seconds


# ============================================================
# START MUJOCO
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("\n====================================")
    print("       HEAL 6-DOF MOTION")
    print("====================================")

    print("Starting position:")
    print(start_deg)

    print("\nTarget position:")
    print(target_deg)

    print("\nRobot is moving...")


    # ========================================================
    # MOVE FROM START TO TARGET
    # ========================================================

    start_time = time.time()

    while viewer.is_running():

        elapsed = time.time() - start_time

        t = elapsed / motion_time

        if t > 1:
            t = 1

        # Smooth interpolation
        s = smooth_motion(t)

        # Calculate current desired position
        desired = start_rad + s * (target_rad - start_rad)

        # Send desired position to motors
        for i in range(6):
            data.ctrl[motor_ids[i]] = desired[i]

        # Simulate
        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()

        time.sleep(model.opt.timestep)

        if t >= 1:
            break


    # ========================================================
    # HOLD TARGET POSITION
    # ========================================================

    print("\nTarget reached.")
    print("Holding position...")

    hold_start = time.time()

    while viewer.is_running():

        elapsed = time.time() - hold_start

        if elapsed >= hold_time:
            break

        # Keep commanding target
        for i in range(6):
            data.ctrl[motor_ids[i]] = target_rad[i]

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)


    # ========================================================
    # RETURN TO START
    # ========================================================

    print("Returning to starting position...")

    return_start = time.time()

    while viewer.is_running():

        elapsed = time.time() - return_start

        t = elapsed / motion_time

        if t > 1:
            t = 1

        s = smooth_motion(t)

        # Target -> Start
        desired = target_rad + s * (start_rad - target_rad)

        for i in range(6):
            data.ctrl[motor_ids[i]] = desired[i]

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

        if t >= 1:
            break


    # ========================================================
    # KEEP ROBOT AT STARTING POSITION
    # ========================================================

    print("\nMotion completed.")
    print("HEAL is now holding the starting position.")

    while viewer.is_running():

        for i in range(6):
            data.ctrl[motor_ids[i]] = start_rad[i]

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)
