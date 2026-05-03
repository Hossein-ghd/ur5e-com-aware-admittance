import sys
import signal
import math
import time
import numpy as np
import logging
import socket
from pathlib import Path
import csv

RUN_NAME = "objA_trial01_corr" #for nominal run use "objA_trial01_nom"
OBJECT_ID = "objA"
OBJECT_ORIENTATION = "ori0"

# Transport compensation mode:
#   "NONE" -> no transport compensation
#   "Z"    -> compensate payload effect only in z
#   "3D"   -> compensate payload effect in all three translational axes
TRANSPORT_COMP_MODE = "Z"
APPLY_TRANSPORT_COMP = (TRANSPORT_COMP_MODE != "NONE")

# Placement correction only in x and y
APPLY_XY_CORR = True #for nominal run use False

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

log_rows = []

sys.path.append("/home/lbeaver/RTDE/RTDE_Python_Client_Library")
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

ROBOT_HOST = "192.168.1.105"
ROBOT_PORT = 30004
GRIPPER_PORT = 30002
config_filename = "control_loop_configuration.xml"
GRASP_TIME = 1.0

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe("state")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

con.get_controller_version()
con.send_output_setup(state_names, state_types)
cmd = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

watchdog.input_int_register_0 = 0
con.send(watchdog)

saved_data1 = np.empty((0, 3))  # force (base)
saved_data2 = np.empty((0, 3))  # tcp position
saved_data3 = np.empty((0, 3))  # position error
saved_data4 = np.empty((0, 3))  # accel (base filtered)
saved_data5 = np.empty((0, 1))  # mass estimate used
saved_data6 = np.empty((0, 3))  # compensated force snapshot
saved_data7 = np.empty((0, 3))  # compensated torque snapshot
saved_data8 = np.empty((0, 3))  # estimated CoM (r) in base
saved_data9 = np.empty((0, 3))  # estimated CoM (r) in tool

payload_mass_locked = 0.0
payload_mass_is_valid = False
mass_est_samples = []

MASS_EST_MIN_SAMPLES = 20
MASS_EST_MAX_SAMPLES = 100
MASS_EST_DEN_THRESH = 2.0
MASS_EST_EXTRA_DELAY = 1.5

MEASURE_POS_TOL = 0.0075
COM_EST_MIN_SAMPLES = 20

def create_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((ROBOT_HOST, GRIPPER_PORT))
    return sock

keep_running = True

def shutdown_handler(sig, frame):
    global keep_running
    global saved_data1, saved_data2, saved_data3, saved_data4, saved_data5
    global saved_data6, saved_data7, saved_data8, saved_data9, log_rows

    watchdog.input_int_register_0 = 0
    con.send(watchdog)
    con.send_pause()
    con.disconnect()
    try:
        sock.close()
    except Exception:
        pass
    keep_running = False

    np.savetxt(DATA_DIR / f"{RUN_NAME}_force.csv", saved_data1, delimiter=",")
    np.savetxt(DATA_DIR / f"{RUN_NAME}_tcp.csv", saved_data2, delimiter=",")
    np.savetxt(DATA_DIR / f"{RUN_NAME}_error.csv", saved_data3, delimiter=",")
    np.savetxt(DATA_DIR / f"{RUN_NAME}_acc.csv", saved_data4, delimiter=",")
    np.savetxt(DATA_DIR / f"{RUN_NAME}_mass.csv", saved_data5, delimiter=",")
    np.savetxt(DATA_DIR / f"{RUN_NAME}_force_comp.csv", saved_data6, delimiter=",")
    np.savetxt(DATA_DIR / f"{RUN_NAME}_torque_comp.csv", saved_data7, delimiter=",")
    np.savetxt(DATA_DIR / f"{RUN_NAME}_com_est.csv", saved_data8, delimiter=",")
    np.savetxt(DATA_DIR / f"{RUN_NAME}_com_est_tool.csv", saved_data9, delimiter=",")

    if len(log_rows) > 0:
        fieldnames = list(log_rows[0].keys())
        with open(DATA_DIR / f"{RUN_NAME}_state_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

def send_gripper_command(sock, command):
    message = f"{command}\n"
    sock.sendall(message.encode("utf-8"))
    try:
        response = sock.recv(1024)
        print(f"Raw response: {response}")
    except socket.timeout:
        print("Gripper command timeout")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")

def close_gripper(sock, con):
    if con.is_connected():
        send_gripper_command(sock, "set_digital_out(8, True)")
        time.sleep(0.5)

def open_gripper(sock, con):
    if con.is_connected():
        send_gripper_command(sock, "set_digital_out(8, False)")
        time.sleep(0.5)

def list_to_setp(setp, velocity, grip):
    for i in range(6):
        setattr(setp, f"input_double_register_{i}", velocity[i])
    setp.input_double_register_6 = grip

def skew(v):
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])

def rodrigues_rotate(v, axis, theta):
    return (
        v * np.cos(theta)
        + np.cross(axis, v) * np.sin(theta)
        + axis * np.dot(axis, v) * (1.0 - np.cos(theta))
    )

def add_vec3(row, name, v):
    row[f"{name}_x"] = float(v[0])
    row[f"{name}_y"] = float(v[1])
    row[f"{name}_z"] = float(v[2])

if not con.send_start():
    sys.exit()

t0 = time.time()
signal.signal(signal.SIGINT, shutdown_handler)

tare = None
acc_tare = 0.0
tau_tare = None
gripper_wt = np.array([0.0, 0.0, -11.45])
grasp = 0
grasp_delay = time.time()

N_SAMPLES = 50
M_SAMPLES = 30
compensated_force = np.zeros((3, N_SAMPLES))
compensated_torque = np.zeros((3, N_SAMPLES))
tool_accelerometer_tf_filtered = np.zeros((3, M_SAMPLES))
counter = 0

F_MIN_NORM = 5.0

COM_AVG_WINDOW = 50
com_tool_samples = []
com_offset_is_valid = False

com_est = np.zeros(3)
com_est_tool = np.zeros(3)
com_est_tool_avg = np.zeros(3)
com_est_avg = np.zeros(3)

measure_wp_idx = 3
pre_place_wp_idx = 4
place_wp_idx = 5
release_wp_idx = 6
STACK_HEIGHT = 0.021
stack_count = 0

waypoints = np.array([
    [-0.10, 0.350, 0.300],   # wp1
    [0.000, 0.685, 0.050],   # wp2
    [0.000, 0.680, -0.013],  # wp3 close
    [0.133, 0.492, 0.2446],  # wp4 measurement
    [-0.30, 0.300, 0.030],   # wp5 pre-place
    [-0.30, 0.300, 0.020],   # wp6 place
    [0.000, 0.400, 0.300],   # wp7 retreat
])

wp_num = 0
waypoint = waypoints[wp_num]
sock = create_socket()

velocity = np.zeros(6)
desired_accel_m1 = np.zeros(3)
desired_velocity_3d = np.zeros(3)

# Disturbance-free reference model kept for checking only.
# The plotting trajectory requested for Fig. 3 is the integrated commanded trajectory below.
ideal_position_3d = None
ideal_velocity_3d = np.zeros(3)
ideal_accel_3d = np.zeros(3)

# Position obtained by integrating the final commanded velocity sent to RTDE.
# This is logged both as cmd_traj_* and ideal_traj_* for compatibility with the current plot file.
cmd_position_3d = None
cmd_velocity_3d = np.zeros(3)

prev_time = time.time()

while keep_running:
    now_time = time.time()
    dt = now_time - prev_time
    prev_time = now_time

    state = con.receive()
    if state is None:
        break

    wp_num_ctrl = wp_num

    wrench_tool = np.array([
        state.ft_raw_wrench[0],
        state.ft_raw_wrench[1],
        state.ft_raw_wrench[2],
        state.ft_raw_wrench[3],
        state.ft_raw_wrench[4],
        state.ft_raw_wrench[5],
    ])

    tcp_pose = np.array([
        state.actual_TCP_pose[0],
        state.actual_TCP_pose[1],
        state.actual_TCP_pose[2],
        state.actual_TCP_pose[3],
        state.actual_TCP_pose[4],
        state.actual_TCP_pose[5],
    ])

    tool_accelerometer = np.array([
        -state.actual_tool_accelerometer[0],
        -state.actual_tool_accelerometer[1],
        -state.actual_tool_accelerometer[2],
    ])

    rotation_vector = tcp_pose[3:]
    theta = np.linalg.norm(rotation_vector)
    rotation_axis = rotation_vector / theta if theta > 1e-9 else rotation_vector

    force_tool = wrench_tool[:3]
    torque_tool = wrench_tool[3:]

    force_data_tf = rodrigues_rotate(force_tool, rotation_axis, theta)
    torque_data_tf = rodrigues_rotate(torque_tool, rotation_axis, theta)

    tool_accelerometer_tf = rodrigues_rotate(tool_accelerometer, rotation_axis, theta)
    gripper_tf = rodrigues_rotate(gripper_wt, rotation_axis, theta)

    if tare is None:
        tare = force_data_tf + gripper_tf
    if tau_tare is None:
        tau_tare = torque_data_tf.copy()

    compensated_force[:, np.mod(counter, N_SAMPLES)] = force_data_tf - tare + gripper_tf
    compensated_force_filtered = np.mean(compensated_force, axis=1)

    compensated_torque[:, np.mod(counter, N_SAMPLES)] = torque_data_tf - tau_tare
    compensated_torque_filtered = np.mean(compensated_torque, axis=1)

    if counter == N_SAMPLES:
        tare = compensated_force_filtered + tare
        tau_tare = compensated_torque_filtered + tau_tare

    tool_accelerometer_tf = tool_accelerometer_tf + np.array([0.0, 0.0, 9.81])
    tool_accelerometer_tf_filtered[:, np.mod(counter, M_SAMPLES)] = tool_accelerometer_tf
    tool_accelerometer_tf_filtered_new = np.mean(tool_accelerometer_tf_filtered, axis=1) - acc_tare
    if counter == M_SAMPLES:
        acc_tare = tool_accelerometer_tf_filtered_new

    # Virtual admittance parameters
    m = 4.0
    k = 300.0
    b = 4.0 * 2.0 * np.sqrt(m * k)

    actual_position = np.array(tcp_pose[:3])
    waypoint_nom = waypoint.copy()
    waypoint_cmd = waypoint.copy()

    if ideal_position_3d is None:
        ideal_position_3d = actual_position.copy()

    # Apply CoM-based placement correction only in x and y
    if APPLY_XY_CORR and (grasp == 100) and (now_time > grasp_delay) and \
       (wp_num_ctrl in [pre_place_wp_idx, place_wp_idx, release_wp_idx]):
        waypoint_cmd[0] = waypoint[0] - com_est_avg[0]
        waypoint_cmd[1] = waypoint[1] - com_est_avg[1]

    # z command is kept as the nominal placement height
    waypoint_cmd[2] = waypoint[2]
    if (grasp == 100) and (now_time > grasp_delay) and \
       (wp_num_ctrl in [pre_place_wp_idx, place_wp_idx, release_wp_idx]):
        waypoint_cmd[2] = waypoint[2] + stack_count * STACK_HEIGHT

    error = waypoint_cmd[:3] - actual_position
    reached_pos = (np.linalg.norm(waypoint_cmd[:3] - actual_position) < MEASURE_POS_TOL)

    gripping_now = (grasp == 100) and (now_time > grasp_delay)
    at_measure_wp = gripping_now and (wp_num_ctrl == measure_wp_idx) and reached_pos

    mass_est_window_active = (
        gripping_now
        and (wp_num_ctrl == measure_wp_idx)
        and (now_time > grasp_delay + MASS_EST_EXTRA_DELAY)
    )

    z_comp_blocked = gripping_now and (wp_num_ctrl in [place_wp_idx, release_wp_idx])

    external_force_3d = compensated_force_filtered[:3].copy()
    payload_force_hat_3d = np.zeros(3)
    residual_force_3d = external_force_3d.copy()

    g_vec = np.array([0.0, 0.0, 9.81])
    acc_minus_g = tool_accelerometer_tf_filtered_new[:3] - g_vec

    den_z = tool_accelerometer_tf_filtered_new[2] - g_vec[2]
    mass_estimation_raw = 0.0
    mass_estimation_used = 0.0

    if abs(den_z) > MASS_EST_DEN_THRESH:
        mass_estimation_raw = compensated_force_filtered[2] / den_z

    # Stage 1: estimate payload mass
    if mass_est_window_active and (not payload_mass_is_valid):
        if abs(den_z) > MASS_EST_DEN_THRESH and np.isfinite(mass_estimation_raw):
            if mass_estimation_raw > 0.0:
                mass_est_samples.append(mass_estimation_raw)
                if len(mass_est_samples) > MASS_EST_MAX_SAMPLES:
                    mass_est_samples.pop(0)

    if mass_est_window_active and (not payload_mass_is_valid):
        if len(mass_est_samples) >= MASS_EST_MIN_SAMPLES:
            payload_mass_locked = float(np.mean(mass_est_samples))
            payload_mass_is_valid = True

    if gripping_now and payload_mass_is_valid:
        mass_estimation_used = payload_mass_locked
    else:
        mass_estimation_used = 0.0

    # Stage 2: use F_exc idea to compensate payload effect in z
    if gripping_now and APPLY_TRANSPORT_COMP and payload_mass_is_valid:
        if TRANSPORT_COMP_MODE == "Z":
            if not z_comp_blocked:
                payload_force_hat_3d[2] = mass_estimation_used * acc_minus_g[2]
        elif TRANSPORT_COMP_MODE == "3D":
            payload_force_hat_3d = mass_estimation_used * acc_minus_g
            if z_comp_blocked:
                payload_force_hat_3d[2] = 0.0

        external_force_3d -= payload_force_hat_3d

    residual_force_3d = external_force_3d.copy()

    desired_accel = (1.0 / m) * (residual_force_3d - b * desired_velocity_3d + k * error)
    desired_velocity_3d += desired_accel * dt
    desired_accel_m1 = desired_accel

    if gripping_now and (now_time > grasp_delay):
        ideal_error = waypoint_cmd[:3] - ideal_position_3d
        ideal_accel_3d = (1.0 / m) * (-b * ideal_velocity_3d + k * ideal_error)
        ideal_velocity_3d += ideal_accel_3d * dt
    else:
        ideal_position_3d = actual_position.copy()
        ideal_velocity_3d[:] = 0.0
        ideal_accel_3d[:] = 0.0

    # Stage 3: estimate CoM from FT and use moving average
    force_comp_tool = rodrigues_rotate(compensated_force_filtered, rotation_axis, -theta)
    torque_comp_tool = rodrigues_rotate(compensated_torque_filtered, rotation_axis, -theta)

    com_est_window_active = (
        gripping_now
        and (wp_num_ctrl == measure_wp_idx)
        and payload_mass_is_valid
        and (now_time > grasp_delay + MASS_EST_EXTRA_DELAY)
    )

    if com_est_window_active and (abs(force_comp_tool[2]) > F_MIN_NORM):
        com_est_tool = np.array([
            -torque_comp_tool[1] / force_comp_tool[2],
             torque_comp_tool[0] / force_comp_tool[2],
             0.0
        ])
        com_est = rodrigues_rotate(com_est_tool, rotation_axis, theta)

        com_tool_samples.append(com_est_tool.copy())
        if len(com_tool_samples) > COM_AVG_WINDOW:
            com_tool_samples.pop(0)

        if len(com_tool_samples) >= COM_EST_MIN_SAMPLES:
            com_est_tool_avg = np.mean(np.vstack(com_tool_samples), axis=0)
            com_est_avg = rodrigues_rotate(com_est_tool_avg, rotation_axis, theta)
            com_offset_is_valid = True
    else:
        com_est_tool = np.zeros(3)
        com_est = np.zeros(3)

    measurement_done = (
        (wp_num_ctrl == measure_wp_idx)
        and reached_pos
        and payload_mass_is_valid
        and com_offset_is_valid
    )

    print(f"Mode={TRANSPORT_COMP_MODE}, comp_on={APPLY_TRANSPORT_COMP}, xy_corr={APPLY_XY_CORR}")
    print(f"wp={wp_num_ctrl}, gripping={gripping_now}, at_measure_wp={at_measure_wp}, measure_done={measurement_done}")
    print(f"Mass raw/used/locked/valid [kg]: {mass_estimation_raw:.3f}, {mass_estimation_used:.3f}, {payload_mass_locked:.3f}, {int(payload_mass_is_valid)}")
    print(f"Mass samples: {len(mass_est_samples)} | CoM samples: {len(com_tool_samples)}")
    print(f"payload_force_hat_3d [N]: {payload_force_hat_3d}")
    print(f"residual_force_3d [N]: {residual_force_3d}")
    print(f"tcp xyz [mm]: {1000.0 * actual_position}")
    print(f"waypoint_cmd xyz [mm]: {1000.0 * waypoint_cmd[:3]}")
    print(f"position error xyz [mm]: {1000.0 * error}")
    print(f"CoM inst tool [mm]: {1000.0 * com_est_tool}")
    print(f"CoM avg tool  [mm]: {1000.0 * com_est_tool_avg}")
    print(f"CoM avg base  [mm]: {1000.0 * com_est_avg}")
    print(f"Applied XY correction [mm]: {np.array([-1000.0 * com_est_avg[0], -1000.0 * com_est_avg[1]])}")
    print("-" * 80)

    vmax = 0.5
    speed_norm = np.linalg.norm(desired_velocity_3d)
    if speed_norm > vmax:
        desired_velocity_3d = desired_velocity_3d / speed_norm * vmax

    ideal_speed_norm = np.linalg.norm(ideal_velocity_3d)
    if ideal_speed_norm > vmax:
        ideal_velocity_3d = ideal_velocity_3d / ideal_speed_norm * vmax

    ideal_position_3d += ideal_velocity_3d * dt

    if wp_num_ctrl == measure_wp_idx and gripping_now:
        reached = measurement_done
    else:
        reached = reached_pos

    if reached:
        wp_num += 1
        if wp_num == len(waypoints):
            wp_num = 0
        waypoint = waypoints[wp_num, :]

        if wp_num == 3:
            grasp = 100
            close_gripper(sock, con)
            grasp_delay = now_time + GRASP_TIME

            payload_mass_locked = 0.0
            payload_mass_is_valid = False
            mass_est_samples = []

            com_tool_samples = []
            com_offset_is_valid = False
            com_est_tool_avg = np.zeros(3)
            com_est_avg = np.zeros(3)
            com_est_tool = np.zeros(3)
            com_est = np.zeros(3)

        elif wp_num == 6:
            grasp = 0
            open_gripper(sock, con)
            grasp_delay = now_time + GRASP_TIME
            stack_count += 1
            print("Placed object count:", stack_count)

            payload_mass_locked = 0.0
            payload_mass_is_valid = False
            mass_est_samples = []

            com_tool_samples = []
            com_offset_is_valid = False
            com_est_tool_avg = np.zeros(3)
            com_est_avg = np.zeros(3)
            com_est_tool = np.zeros(3)
            com_est = np.zeros(3)

    if now_time <= grasp_delay:
        desired_velocity_3d = np.zeros(3)
        print("Waiting to grasp...")

    velocity[:3] = desired_velocity_3d
    velocity[3:6] = 0.0

    # Integrate the final commanded velocity sent to RTDE.
    # This gives the commanded TCP trajectory used for plotting.
    if cmd_position_3d is None:
        cmd_position_3d = actual_position.copy()

    if gripping_now and (now_time > grasp_delay):
        cmd_position_3d += velocity[:3] * dt
        cmd_velocity_3d = velocity[:3].copy()
    else:
        cmd_position_3d = actual_position.copy()
        cmd_velocity_3d[:] = 0.0

    accelerometer_6 = np.zeros(6)
    accelerometer_6[:3] = desired_accel_m1

    list_to_setp(cmd, velocity, grasp)
    con.send(cmd)
    con.send(watchdog)

    rel_time = now_time - t0

    row = {
        "run_name": RUN_NAME,
        "object_id": OBJECT_ID,
        "object_orientation": OBJECT_ORIENTATION,
        "sample": int(counter),
        "t": float(rel_time),
        "dt": float(dt),
        "wp_num": int(wp_num_ctrl),
        "grasp": int(grasp),
        "gripping_now": int(gripping_now),
        "transport_comp_enabled": int(APPLY_TRANSPORT_COMP),
        "transport_comp_mode": TRANSPORT_COMP_MODE,
        "apply_xy_corr": int(APPLY_XY_CORR),
        "stack_count": int(stack_count),
        "measure_stage": int(at_measure_wp),
        "measure_done": int(measurement_done),
        "mass_est_window_active": int(mass_est_window_active),
        "com_est_window_active": int(com_est_window_active),
        "mass_sample_count": int(len(mass_est_samples)),
        "com_sample_count": int(len(com_tool_samples)),
        "pre_place_stage": int(wp_num_ctrl == pre_place_wp_idx),
        "place_stage": int(wp_num_ctrl == place_wp_idx),
        "release_stage": int(wp_num_ctrl == release_wp_idx),
        "mass_estimation_active": float(mass_estimation_used),
        "mass_estimation_raw": float(mass_estimation_raw),
        "payload_mass_locked": float(payload_mass_locked),
        "payload_mass_is_valid": int(payload_mass_is_valid),
        "speed_cmd_norm": float(np.linalg.norm(desired_velocity_3d)),
    }

    add_vec3(row, "tcp", actual_position)
    add_vec3(row, "waypoint_nom", waypoint_nom[:3])
    add_vec3(row, "waypoint_cmd", waypoint_cmd[:3])
    add_vec3(row, "error", error)
    add_vec3(row, "desired_vel", desired_velocity_3d)
    add_vec3(row, "desired_acc", desired_accel)

    # Keep ideal_traj_* for compatibility with the existing plot file.
    # Here ideal_traj_* means the integrated commanded trajectory.
    add_vec3(row, "ideal_traj", cmd_position_3d)
    add_vec3(row, "ideal_vel", cmd_velocity_3d)
    add_vec3(row, "ideal_acc", desired_accel)

    # Also save explicit names to avoid ambiguity in future plots.
    add_vec3(row, "cmd_traj", cmd_position_3d)
    add_vec3(row, "cmd_vel", cmd_velocity_3d)

    # Save the disturbance-free admittance reference separately for checking.
    add_vec3(row, "ref_traj", ideal_position_3d)
    add_vec3(row, "ref_vel", ideal_velocity_3d)
    add_vec3(row, "ref_acc", ideal_accel_3d)

    add_vec3(row, "force_comp_base", compensated_force_filtered)
    add_vec3(row, "payload_force_hat_base", payload_force_hat_3d)
    add_vec3(row, "force_residual_base", residual_force_3d)
    add_vec3(row, "torque_comp_base", compensated_torque_filtered)
    add_vec3(row, "acc_filt_base", tool_accelerometer_tf_filtered_new)
    add_vec3(row, "com_est_base", com_est)
    add_vec3(row, "com_est_tool", com_est_tool)
    add_vec3(row, "com_est_avg_base", com_est_avg)
    add_vec3(row, "com_est_avg_tool", com_est_tool_avg)

    log_rows.append(row)
    counter += 1

    saved_data1 = np.vstack([saved_data1, force_data_tf.reshape(1, 3)])
    saved_data2 = np.vstack([saved_data2, tcp_pose[:3].reshape(1, 3)])
    saved_data3 = np.vstack([saved_data3, error.reshape(1, 3)])
    saved_data4 = np.vstack([saved_data4, tool_accelerometer_tf_filtered_new.reshape(1, 3)])
    saved_data5 = np.vstack([saved_data5, np.array([[mass_estimation_used]])])
    saved_data6 = np.vstack([saved_data6, compensated_force_filtered.reshape(1, 3)])
    saved_data7 = np.vstack([saved_data7, compensated_torque_filtered.reshape(1, 3)])
    saved_data8 = np.vstack([saved_data8, com_est.reshape(1, 3)])
    saved_data9 = np.vstack([saved_data9, com_est_tool.reshape(1, 3)])