import mujoco
import mujoco.viewer
import numpy as np
import time


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "7DOF_FRANKA_Manipulator.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# ACTUATORS
# ============================================================

motor_names = [
    "motor1",
    "motor2",
    "motor3",
    "motor4",
    "motor5",
    "motor6",
    "motor7"
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
# STARTING POSITION
# ============================================================

start_deg = np.array([
    0,
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
    60,
    -60,
    20,
    30,
    0,
    45
], dtype=float)


start_rad = np.deg2rad(start_deg)
target_rad = np.deg2rad(target_deg)


# ============================================================
# INITIALIZE ROBOT
# ============================================================

data.qpos[:7] = start_rad
data.qvel[:7] = 0

for i in range(7):

    data.ctrl[motor_ids[i]] = start_rad[i]

mujoco.mj_forward(model, data)


# ============================================================
# SMOOTH TRAJECTORY
# ============================================================

def smooth_step(t):

    return 0.5 * (
        1.0 - np.cos(np.pi * t)
    )


# ============================================================
# MOTION PARAMETERS
# ============================================================

motion_time = 5.0
hold_time = 2.0


# ============================================================
# START SIMULATION
# ============================================================

with mujoco.viewer.launch_passive(
        model,
        data) as viewer:

    print()
    print("======================================")
    print("       7-DOF HEAL MANIPULATOR")
    print("======================================")
    print()
    print("Starting position:")
    print(start_deg)
    print()
    print("Target position:")
    print(target_deg)
    print()
    print("Moving...")
    print()


    # ========================================================
    # START -> TARGET
    # ========================================================

    start_time = time.time()

    while viewer.is_running():

        elapsed = time.time() - start_time

        t = elapsed / motion_time

        t = min(t, 1.0)

        s = smooth_step(t)

        desired = (
            start_rad
            + s * (target_rad - start_rad)
        )

        for i in range(7):

            data.ctrl[motor_ids[i]] = desired[i]

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

        if t >= 1.0:

            break


    # ========================================================
    # HOLD TARGET
    # ========================================================

    print("Target reached.")
    print("Holding position...")

    hold_start = time.time()

    while viewer.is_running():

        elapsed = time.time() - hold_start

        if elapsed >= hold_time:

            break

        for i in range(7):

            data.ctrl[motor_ids[i]] = target_rad[i]

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)


    # ========================================================
    # TARGET -> START
    # ========================================================

    print("Returning to starting position...")

    return_start = time.time()

    while viewer.is_running():

        elapsed = time.time() - return_start

        t = elapsed / motion_time

        t = min(t, 1.0)

        s = smooth_step(t)

        desired = (
            target_rad
            + s * (start_rad - target_rad)
        )

        for i in range(7):

            data.ctrl[motor_ids[i]] = desired[i]

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

        if t >= 1.0:

            break


    # ========================================================
    # HOLD START POSITION
    # ========================================================

    print("Motion completed.")
    print("Robot is holding the starting position.")

    while viewer.is_running():

        for i in range(7):

            data.ctrl[motor_ids[i]] = start_rad[i]

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)
