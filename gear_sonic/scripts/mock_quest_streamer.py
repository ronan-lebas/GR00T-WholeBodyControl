import argparse
import zmq
import time
import math
import numpy as np
import sys
import select
import termios
import tty
from pathlib import Path

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

# Per-joint [lower, upper] travel limits (radians), taken verbatim from the DEX3
# MJCF (model_data/g1/with_dex3/g1_29dof_with_hand.xml). Joint order matches the
# wire format: [thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1].
# Left and right differ because the hands use a mirrored joint convention, so the
# tables already encode the correct per-hand signs (no extra mirroring needed).
# The right thumb_0 (metacarpal) is the exception that keeps the left's sign: its
# base link is mounted as a geometric reflection, so the same joint sign already
# yields mirror-symmetric world motion. These limits bake that in already.
DEX3_LIMITS_LEFT = [
    (-1.0472, 1.0472),    # thumb_0
    (-0.724312, 1.0472),  # thumb_1
    (0.0, 1.74533),       # thumb_2
    (-1.5708, 0.0),       # index_0
    (-1.74533, 0.0),      # index_1
    (-1.5708, 0.0),       # middle_0
    (-1.74533, 0.0),      # middle_1
]
DEX3_LIMITS_RIGHT = [
    (-1.0472, 1.0472),    # thumb_0
    (-1.0472, 0.724312),  # thumb_1
    (-1.74533, 0.0),      # thumb_2
    (0.0, 1.5708),        # index_0
    (0.0, 1.74533),       # index_1
    (0.0, 1.5708),        # middle_0
    (0.0, 1.74533),       # middle_1
]

# ---------------------------------------------------------------------------
# VR 3-point wrist/head targets (root-normalized frame; X-forward, Y-left, Z-up).
# Layout is 9 values: [L wrist xyz, R wrist xyz, head xyz], matching the wire order.
# ---------------------------------------------------------------------------

# Steady-state targets streamed once teleop is fully engaged.
VR_3PT_NOMINAL = [
    0.3,  0.2, 0.3,   # Left Wrist
    0.3, -0.2, 0.3,   # Right Wrist
    0.0,  0.0, 0.4,   # Head/Neck
]

# Pose to ramp *from* when the controller is first entered, so the arms ease
# into the nominal pose instead of snapping to it. This mock is publish-only and
# has no robot-state feedback, so this is an *assumed* rest pose (arms lowered
# and pulled in toward the body). Set it close to the robot's actual pose at
# calibration to keep the very start of the ramp gentle. Head is kept at nominal
# (only the arms ramp).
VR_3PT_START = [
    0.15,  0.2, -0.2,  # Left Wrist
    0.15, -0.2, -0.2,  # Right Wrist
    0.0,   0.0, 0.4,  # Head/Neck (unchanged; does not ramp)
]

# Seconds to ramp from VR_3PT_START to VR_3PT_NOMINAL after entering VR_3PT mode.
RAMP_DURATION_S = 3.0

# Arm Z sine animation (toggled with 'a'): amplitude (m) and angular frequency (rad/s).
ARM_ANIM_AMP = 0.1
ARM_ANIM_OMEGA = 3.0
# Seconds to smoothly decay the arm sine to 0 after it is toggled off (fast but not a snap).
ARM_ANIM_FADE_OUT_S = 0.4


def _smoothstep(x: float) -> float:
    """Clamp x to [0, 1] and apply cubic smoothstep easing (0->0, 1->1, zero slope at ends)."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def retarget_joints_to_wire(joints_normalized: np.ndarray) -> list[float]:
    """Convert the 6 normalized [0,1] joints from video_retarget to the 7-element
    wire format. Video retarget is BrainCo-only (rejected for DEX3 in main()).

    BrainCo uses normalized [0,1] joints — the deploy/sim side maps them onto each
    joint's range — so both hands share the same wire vector; the 7th slot is
    padding ignored by the firmware.
    """
    return list(joints_normalized) + [0.0]


def make_demo_joints(t: float, hand_type: str, is_left: bool = True) -> list[float]:
    """Return a 7-element joint list for time t — a per-joint range-of-motion demo.

    Every joint follows its own cosine sweep across its FULL travel, each with a
    distinct phase offset so the joints move independently.

    hand_type: 'brainco' or 'dex3'
    is_left:   selects the hand's per-joint limits/convention (DEX3 only).

    BrainCo: 6 normalized values [0,1] + 1 padding zero. The sim/deploy side maps
             [0,1] onto each joint's [lower, upper] range, so the normalized sweep
             already spans full travel and is identical for both hands (``is_left``
             ignored).
    DEX3:    7 joint angles in radians, each swept over its own [lo, hi] limit from
             DEX3_LIMITS_{LEFT,RIGHT}. The per-hand tables encode the mirrored
             joint convention, so no extra sign handling is needed here.
    """
    omega = math.pi / 3.0  # ~6 s period

    if hand_type == "brainco":
        # Fingers: Thumb, Thumb_aux, Index, Middle, Ring, Pinky
        # Evenly spaced phase offsets over one full period (2π), ~6 s cycle each.
        joints = [
            0.5 - 0.5 * math.cos(omega * t + i * 2.0 * math.pi / 6.0)
            for i in range(6)
        ]
        return joints + [0.0]  # 7th slot: unused, padded with 0
    else:  # dex3
        limits = DEX3_LIMITS_LEFT if is_left else DEX3_LIMITS_RIGHT
        n = len(limits)
        out = []
        for i, (lo, hi) in enumerate(limits):
            phase = i * 2.0 * math.pi / n
            s = 0.5 - 0.5 * math.cos(omega * t + phase)  # [0, 1]
            out.append(lo + (hi - lo) * s)  # sweep lo <-> hi
        return out


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

    if args.video_retarget and args.hand != "brainco":
        raise SystemExit(
            "--video-retarget is only supported with --hand brainco. The retargeting "
            "module outputs BrainCo-style normalized [0,1] joints and has no DEX3 "
            "mapping, so it is not expected to work with dex3. Use the keyboard "
            "cosine animation ('f' toggle) for dex3 instead."
        )

    if args.video_retarget:
        _demos_path = Path(__file__).resolve().parents[2] / "third_party" / "brainco-retargeting" / "demos"
        sys.path.insert(0, str(_demos_path))
        import retarget_streaming
        retarget_streaming.start(hand="right")
        print("retarget_streaming started — camera GUI should be open.")

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
    arm_anim_start_t = 0.0   # loop-clock time at which the arm sine was last (re)started
    arm_anim_fade_t = 0.0    # loop-clock time at which the fade-out began
    arm_anim_fading = False  # True while smoothly decaying the sine to 0 after deactivation
    print("Keyboard controls:")
    print("  c  - Toggle data collection (Left Grip + A)")
    print("  x  - Toggle data abort (Left Grip + B)")
    retarget_src = "video_retarget module" if args.video_retarget else "per-joint full-range cosine sweep, ~6 s period"
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
                    if arm_z_animation_enabled:
                        # Restart the sine clock so the animation begins at a zero
                        # offset (from the deactivated position) rather than jumping.
                        arm_anim_start_t = t
                        arm_anim_fading = False
                    else:
                        # Begin a short, smooth fade-out from the current offset
                        # instead of snapping back to center.
                        arm_anim_fade_t = t
                        arm_anim_fading = True
                    state = "ENABLED" if arm_z_animation_enabled else "DISABLED"
                    print(f"Pressed 'a': Arm Z animation {state}")

            # 3. Generate Mocked 3-Point VR Data
            # Coordinate System: X-forward, Y-left, Z-up
            # Ramp the wrist/head targets from VR_3PT_START to VR_3PT_NOMINAL over
            # the first RAMP_DURATION_S seconds so the arms ease in instead of
            # snapping when the controller is entered.
            ramp = _smoothstep(t / RAMP_DURATION_S)
            vr_3pt_pos = [
                start + ramp * (nominal - start)
                for start, nominal in zip(VR_3PT_START, VR_3PT_NOMINAL)
            ]

            # Optional arm Z sine, added on top of the ramped base. Its clock
            # restarts from 0 on each activation so the motion always eases out of
            # the current (neutral) position. When toggled off it fades out over
            # ARM_ANIM_FADE_OUT_S: the phase keeps running (velocity stays
            # continuous, no jerk) while the amplitude decays smoothly to 0, so it
            # stops quickly without snapping. Left/right are anti-phase.
            env = None
            if arm_z_animation_enabled:
                env = 1.0
            elif arm_anim_fading:
                frac = (t - arm_anim_fade_t) / ARM_ANIM_FADE_OUT_S
                if frac >= 1.0:
                    arm_anim_fading = False
                else:
                    env = 1.0 - _smoothstep(frac)
            if env is not None:
                off = env * ARM_ANIM_AMP * math.sin(ARM_ANIM_OMEGA * (t - arm_anim_start_t))
                vr_3pt_pos[2] += off   # Left wrist Z
                vr_3pt_pos[5] -= off   # Right wrist Z

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
                    raw = retarget_streaming.get_joints()  # np.ndarray (6,) in [0, 1]
                    # BrainCo only: both hands share the same normalized wire vector.
                    wire = retarget_joints_to_wire(raw)
                    left_hand_joints  = wire
                    right_hand_joints = wire
                else:
                    left_hand_joints  = make_demo_joints(t, args.hand, is_left=True)
                    right_hand_joints = make_demo_joints(t, args.hand, is_left=False)

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
        if args.video_retarget:
            retarget_streaming.stop()
        socket.close()
        context.term()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    main()
