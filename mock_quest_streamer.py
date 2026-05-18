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

def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    port = 5556
    socket.bind(f"tcp://*:{port}")
    
    print(f"🚀 Meta Quest Mock Streamer binding to port {port}...")
    print("Waiting 3 seconds for C++ subscriber to connect...")
    time.sleep(3.0)

    # 1. Emulate pressing A+B+X+Y to START the policy
    print("🟢 Sending START command (Emulating A+B+X+Y)...")
    socket.send(build_command_message(start=True, stop=False, planner=True))
    
    # Give the robot a moment to run its CALIB_FULL routine
    time.sleep(1.0) 

    # 2. Emulate entering VR_3PT Mode (Left Stick Click)
    print("🥽 Entering VR_3PT Mode...")
    stream_mode = 5  # 5 = StreamMode.PLANNER_VR_3PT
    
    t0 = time.time()
    
    try:
        while True:
            t = time.time() - t0
            
            toggle_dc = False
            toggle_da = False
            if is_data():
                c = sys.stdin.read(1)
                if c == 'c':
                    print("⌨️  Pressed 'c': Toggling data collection (Emulating Left Grip + A)")
                    toggle_dc = True
                elif c == 'x':
                    print("⌨️  Pressed 'x': Toggling data abort (Emulating Left Grip + B)")
                    toggle_da = True

            # 3. Generate Mocked 3-Point VR Data
            # Moving the wrists up and down in a sine wave.
            # Coordinate System: X-forward, Y-left, Z-up
            left_z = 0.3 + 0.2 * math.sin(t * 3.0)
            right_z = 0.3 + 0.2 * math.cos(t * 3.0)
            
            vr_3pt_pos = [
                0.3,  0.2, left_z,   # Left Wrist (X, Y, Z)
                0.3, -0.2, right_z,  # Right Wrist (X, Y, Z)
                0.0,  0.0, 0.4       # Head/Neck (X, Y, Z) 
            ]
            
            vr_3pt_quat = [
                1.0, 0.0, 0.0, 0.0,  # Left Wrist (W, X, Y, Z)
                1.0, 0.0, 0.0, 0.0,  # Right Wrist (W, X, Y, Z)
                1.0, 0.0, 0.0, 0.0   # Head (W, X, Y, Z)
            ]

            # 4. Send the Manager State (topic: "manager_state")
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

            # 5. Send the Teleop Data (topic: "planner")
            # This feeds our 3 spatial points into the robot's IK solver.
            planner_msg = build_planner_message(
                mode=0,                  # LocomotionMode.IDLE = 0
                movement=[0.0, 0.0, 0.0], # Joystick movement
                facing=[1.0, 0.0, 0.0],   # Joystick yaw
                speed=-1.0,
                height=-1.0,
                upper_body_position=None,
                left_hand_position=None,  # Replace with [7] array when you want finger tracking
                right_hand_position=None, # Replace with [7] array when you want finger tracking
                vr_3pt_position=vr_3pt_pos,
                vr_3pt_orientation=vr_3pt_quat,
                vr_3pt_compliance=None,
            )
            socket.send(planner_msg)
            
            # Stream at 50Hz
            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\n🛑 Sending STOP command (Emulating A+B+X+Y emergency stop)...")
        socket.send(build_command_message(start=False, stop=True, planner=True))
        time.sleep(0.1)
    finally:
        socket.close()
        context.term()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    main()
