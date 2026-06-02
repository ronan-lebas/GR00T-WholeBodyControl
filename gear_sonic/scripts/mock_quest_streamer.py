import argparse
import zmq
import time
import math
import numpy as np
import sys
import select
import termios
import tty

# We import the exact message builders
# so the ZMQ serialization perfectly matches what the C++ code expects.
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

# Wire format always sends 7 values for hand joints (both hand types share the same
# field shape so the C++ receiver does not need to be recompiled to parse them).
# For BrainCo (6 DOF): the first 6 values are meaningful [0=open, 1=closed];
#                       the 7th slot is padded with 0.0 and ignored by the firmware.
# For DEX3 (7 DOF):    all 7 values are joint angles in radians (0.0 = open).
WIRE_HAND_DOF = 7


def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def retarget_joints_to_wire(joints_normalized: np.ndarray, hand_type: str) -> list[float]:
    """Convert 6 normalized [0,1] joints from video_retarget to the 7-element wire format.

    For brainco: append one padding zero (7th slot unused by firmware).
    For dex3: scale each normalized value by the per-joint close limit.
    """
    j = joints_normalized  # shape (6,)
    if hand_type == "brainco":
        return list(j) + [0.0]
    else:  # dex3
        dex3_close = [1.05, 1.05, 0.0, -1.57, -1.75, -1.57, -1.75]
        # Map the first 6 normalized values; 7th slot mirrors j[5] (pinky)
        normalized_7 = list(j) + [j[5]]
        return [n * v for n, v in zip(normalized_7, dex3_close)]


def make_grasp_joints(t: float, hand_type: str) -> list[float]:
    """Return a 7-element joint list for time t.

    hand_type: 'brainco' or 'dex3'

    BrainCo: 6 normalized values [0,1] + 1 padding zero.
             Each finger gets its own phase offset so they move independently,
             clearly demonstrating per-finger control.
    DEX3:    7 joint angles in radians, all fingers in sync.
    """
    if hand_type == "brainco":
        # Fingers: Thumb, Thumb_aux, Index, Middle, Ring, Pinky
        # Evenly spaced phase offsets over one full period (2π), ~6 s cycle each.
        joints = [
            0.5 - 0.5 * math.cos(t * math.pi / 3.0 + i * 2.0 * math.pi / 6.0)
            for i in range(6)
        ]
        return joints + [0.0]  # 7th slot: unused, padded with 0
    else:  # dex3
        # DEX3 joint limits (left hand, close direction):
        #   j0 thumb:  0 -> 1.05 rad
        #   j1 thumb2: 0 -> 1.05 rad
        #   j2 thumb3: 1.75 -> 0 rad (already open at 1.75, close toward 0)
        #   j3..j6 fingers: 0 -> -1.57 / -1.75 rad (close in negative direction)
        grasp = 0.5 - 0.5 * math.cos(t * math.pi / 3.0)  # [0, 1], period ~6 s
        dex3_close = [1.05, 1.05, 0.0, -1.57, -1.75, -1.57, -1.75]
        return [grasp * v for v in dex3_close]


def main():
    parser = argparse.ArgumentParser(description="Mock Meta Quest streamer for testing teleop.")
    parser.add_argument(
        "--hand",
        choices=["dex3", "brainco"],
        default="dex3",
        help="Hand type: 'dex3' (7 DOF, joint angles in rad) or 'brainco' (6 DOF, normalized [0,1]). Default: dex3",
    )
    parser.add_argument(
        "--video-retarget",
        action="store_true",
        help=(
            "Use video_retarget module for finger joints instead of the cosine animation. "
            "Opens a camera GUI and streams live hand detection. "
            "Implies finger tracking; press 'f' still toggles sending the joints."
        ),
    )
    args = parser.parse_args()

    if args.video_retarget:
        import video_retarget
        video_retarget.start(hand="right")
        print("video_retarget started — camera GUI should be open.")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    port = 5556
    socket.bind(f"tcp://*:{port}")

    print(f"Meta Quest Mock Streamer binding to port {port} (hand={args.hand})...")
    print("Waiting 3 seconds for C++ subscriber to connect...")
    time.sleep(3.0)

    # 1. Emulate pressing A+B+X+Y to START the policy
    print("Sending START command (Emulating A+B+X+Y)...")
    socket.send(build_command_message(start=True, stop=False, planner=True))

    # Give the robot a moment to run its CALIB_FULL routine
    time.sleep(1.0)

    # 2. Emulate entering VR_3PT Mode (Left Stick Click)
    print("Entering VR_3PT Mode...")
    stream_mode = 5  # 5 = StreamMode.PLANNER_VR_3PT

    finger_tracking_enabled = False
    arm_z_animation_enabled = False
    print("Keyboard controls:")
    print("  c  - Toggle data collection (Left Grip + A)")
    print("  x  - Toggle data abort (Left Grip + B)")
    retarget_src = "video_retarget module" if args.video_retarget else "slow open/close sine wave, ~6 s period"
    print(f"  f  - Toggle finger tracking ({retarget_src})")
    print("  a  - Toggle arm Z animation (wrists move up/down in sine wave)")
    if args.hand == "brainco":
        print("       BrainCo: 6 DOF, values [0.0=open .. 1.0=closed], 7th slot padded")
    else:
        print("       DEX3: 7 DOF joint angles in radians, 0.0=open")

    t0 = time.time()

    try:
        while True:
            t = time.time() - t0

            toggle_dc = False
            toggle_da = False
            if is_data():
                c = sys.stdin.read(1)
                if c == 'c':
                    print("Pressed 'c': Toggling data collection (Emulating Left Grip + A)")
                    toggle_dc = True
                elif c == 'x':
                    print("Pressed 'x': Toggling data abort (Emulating Left Grip + B)")
                    toggle_da = True
                elif c == 'f':
                    finger_tracking_enabled = not finger_tracking_enabled
                    state = "ENABLED" if finger_tracking_enabled else "DISABLED"
                    print(f"Pressed 'f': Finger tracking {state}")
                elif c == 'a':
                    arm_z_animation_enabled = not arm_z_animation_enabled
                    state = "ENABLED" if arm_z_animation_enabled else "DISABLED"
                    print(f"Pressed 'a': Arm Z animation {state}")

            # 3. Generate Mocked 3-Point VR Data
            # Coordinate System: X-forward, Y-left, Z-up
            left_z  = 0.3 + (0.2 * math.sin(t * 3.0) if arm_z_animation_enabled else 0.0)
            right_z = 0.3 + (0.2 * math.cos(t * 3.0) if arm_z_animation_enabled else 0.0)

            vr_3pt_pos = [
                0.3,  0.2, left_z,   # Left Wrist (X, Y, Z)
                0.3, -0.2, right_z,  # Right Wrist (X, Y, Z)
                0.0,  0.0, 0.4,      # Head/Neck (X, Y, Z)
            ]

            vr_3pt_quat = [
                1.0, 0.0, 0.0, 0.0,  # Left Wrist (W, X, Y, Z)
                1.0, 0.0, 0.0, 0.0,  # Right Wrist (W, X, Y, Z)
                1.0, 0.0, 0.0, 0.0,  # Head (W, X, Y, Z)
            ]

            # 4. Compute finger joint targets when tracking is enabled.
            left_hand_joints = None
            right_hand_joints = None
            if finger_tracking_enabled:
                if args.video_retarget:
                    raw = video_retarget.get_joints()  # np.ndarray (6,) in [0, 1]
                    wire = retarget_joints_to_wire(raw, args.hand)
                    left_hand_joints  = wire
                    right_hand_joints = wire
                else:
                    left_hand_joints  = make_grasp_joints(t, args.hand)
                    right_hand_joints = make_grasp_joints(t, args.hand)

            # 5. Send the Manager State (topic: "manager_state")
            # This tells the C++ state machine that we are staying in VR_3PT mode.
            manager_state_msg = pack_pose_message(
                {
                    "stream_mode": np.array([stream_mode], dtype=np.int32),
                    "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                    "toggle_data_abort": np.array([toggle_da], dtype=bool),
                },
                topic="manager_state",
            )
            socket.send(manager_state_msg)

            # 6. Send the Teleop Data (topic: "planner")
            # This feeds our 3 spatial points into the robot's IK solver,
            # and optionally the finger joint targets (left_hand_joints /
            # right_hand_joints fields in the wire format).
            planner_msg = build_planner_message(
                mode=0,                     # LocomotionMode.IDLE = 0
                movement=[0.0, 0.0, 0.0],   # Joystick movement
                facing=[1.0, 0.0, 0.0],     # Joystick yaw
                speed=-1.0,
                height=-1.0,
                upper_body_position=None,
                left_hand_position=left_hand_joints,
                right_hand_position=right_hand_joints,
                vr_3pt_position=vr_3pt_pos,
                vr_3pt_orientation=vr_3pt_quat,
                vr_3pt_compliance=None,
            )
            socket.send(planner_msg)

            # Stream at 50Hz
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nSending STOP command (Emulating A+B+X+Y emergency stop)...")
        socket.send(build_command_message(start=False, stop=True, planner=True))
        time.sleep(0.1)
    finally:
        socket.close()
        context.term()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    main()
