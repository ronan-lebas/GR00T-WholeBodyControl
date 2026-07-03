# G1 + BrainCo Hands + Meta Quest — Real-Robot Deployment Report

This report explains, end to end, how to deploy the SONIC whole-body controller
on a **real Unitree G1** equipped with **BrainCo Revo2 hands**, teleoperated
from a **Meta Quest 3**, using the modified codebase in this fork.

It is written against the changes introduced from the commit
**`3e8d73f` "moved default g1 to hand-specific folder"** up to `HEAD`
(`9bd0f71`). The emphasis, per the brief, is on the two hard parts:

1. **The BrainCo hands** (a serial→DDS bridge that the C++ control stack does
   *not* build or own, plus a compile-time switch inside the deploy binary).
2. **Deploying the C++ control stack** (`g1_deploy_onnx_ref`) onto the robot's
   on-board compute.

The Quest side is covered more briefly (it is largely turn-key), and points to
`docs/quest_manager_docs.md` for the deep dive.

---

## 0. TL;DR — the bring-up order

On the robot's on-board compute (the Jetson "PC2"), in this order:

1. **BrainCo serial bridge must be live first.**
   `brainco_hand_service` publishes/subscribes the DDS topics
   `rt/brainco/{left,right}/{cmd,state}`. If it is not running, the deploy
   binary's hands silently do nothing (no state feedback, no motion). Run it as
   a systemd service so it is always up.
2. **Confirm the deploy binary is compiled for BrainCo.**
   `hand_config.hpp` must have `#define USE_BRAINCO_HANDS 1` (it already does in
   this fork). This is a *compile-time* switch — changing it requires a rebuild.
3. **Build + launch the control stack** with `gear_sonic_deploy/deploy.sh real`.
   This builds `g1_deploy_onnx_ref` and runs it with the release policy,
   planner, and `--input-type manager` (ZMQ) so the Quest manager can drive it.
4. **Start the Quest relay + manager** (on the dev PC or on-board, see Part C),
   put the operator in the rest pose, and press `s` to start.

If any one of those is wrong, the most common symptoms are: arms move but hands
are frozen (step 1/2), or the robot never leaves `WAIT_FOR_CONTROL` (step 3/4).

---

## 1. System architecture & data flow

There are **four cooperating processes** plus the robot firmware. Three are new
or heavily modified in this fork.

```
  ┌─────────────┐   TCP:10000   ┌───────────────────────────┐
  │  Meta Quest │──────────────▶│  quest_relay (Docker)      │
  │  (Unity app)│  ROS-TCP      │  ros_tcp_endpoint          │
  └─────────────┘  Connector    │  + relay.py (ROS1→ZMQ)     │
                                 └─────────────┬─────────────┘
                                   ZMQ PUB "quest_data"
                                   msgpack, port 5559
                                               │
                                               ▼
                          ┌────────────────────────────────────┐
                          │ quest_manager_thread_server.py      │
                          │  (Python, .venv_teleop)             │
                          │  3-pt VR tracking + finger retarget │
                          └───────┬───────────────────▲────────┘
            ZMQ PUB "command"/    │                   │ ZMQ SUB "g1_debug"
            "planner"/            │                   │ port 5557 (feedback:
            "manager_state"       │                   │ measured joints for
            port 5556             ▼                   │ recalibration)
                          ┌────────────────────────────────────┐
                          │ g1_deploy_onnx_ref  (C++ binary)    │
                          │  TensorRT policy + planner          │
                          │  --input-type manager (ZMQ)         │
                          │  --output-type all (ZMQ 5557 + ROS2)│
                          └───────┬───────────────────┬────────┘
              Unitree DDS         │                   │  Unitree DDS
              rt/lowcmd, rt/lowstate (29-DOF body)    │  rt/brainco/{l,r}/cmd
              over 192.168.123.x  │                   │  rt/brainco/{l,r}/state
                                  ▼                   ▼
                          ┌──────────────┐   ┌────────────────────────┐
                          │ G1 firmware  │   │ brainco_hand_service    │
                          │ (body motors,│   │ (C++, DDS↔RS485/Modbus) │
                          │  IMU)        │   │  → BrainCo Revo2 hands  │
                          └──────────────┘   └────────────────────────┘
```

Key consequences of this topology:

- **The WBC policy is hand-agnostic.** `g1_deploy_onnx_ref` runs the SONIC
  ONNX/TensorRT policy that controls the **29 body DOF only**. Finger commands
  are a *separate* stream that is forwarded straight to the hand driver. The
  same `policy/release` checkpoint works for Dex3 or BrainCo — only the hand
  driver (and the 6-vs-7 motor bookkeeping) changes.
- **Two independent DDS planes** share the robot network: the body
  (`rt/lowcmd`/`rt/lowstate`, owned by Unitree firmware) and the hands
  (`rt/brainco/*`, owned by `brainco_hand_service`). Both must use the same
  network interface / DDS domain. The deploy binary calls
  `ChannelFactory::Init(0, <iface>)` once and the hand driver reuses it.
- **The Quest path is pure ZMQ from the manager onward.** The deploy binary
  has no idea a Quest exists — it just consumes `planner`/`command` ZMQ
  messages. You can therefore test the whole control stack with the mock
  streamer or a replay file, no headset needed.

### Compute layout

| Role | Typical host | Runs |
|---|---|---|
| **On-board / "PC2"** (Jetson Orin NX, `aarch64`) | inside the G1 | `brainco_hand_service`, `g1_deploy_onnx_ref`. Optionally the relay + manager too. |
| **Dev PC** (`x86_64`, optional) | your laptop/workstation | `quest_relay` Docker + `quest_manager_thread_server.py`, if you prefer to keep teleop off-board. |

Either layout works because everything between the manager and the deploy
binary is ZMQ over TCP (`--zmq-host`). The simplest and lowest-latency option is
to run **everything on-board**, but you can also run the heavy/teleop pieces on
your own computer. This choice recurs throughout the report, so read the
networking model below once.

### Networking: where to run the deploy stack (two ways)

There are **two distinct networks/links** in play, and they answer two different
questions:

**(a) The robot control link — DDS, must be on `192.168.123.x`.**
The G1's body firmware (motors + IMU) and `brainco_hand_service` speak **Unitree
DDS** on the robot's internal LAN, subnet **`192.168.123.x`**, DDS domain `0`.
**Whatever process runs `g1_deploy_onnx_ref` must have a network interface on
that subnet**, because that is how it sends `rt/lowcmd` and reads `rt/lowstate`.
This is the link `deploy.sh real` auto-detects.

- **Way 1 — deploy on the robot (on-board Jetson / "PC2").** The Jetson is
  already wired to `192.168.123.x` internally. Nothing to set up; `deploy.sh
  real` finds the interface. Lowest latency, no extra cabling, recommended for
  real runs.
- **Way 2 — deploy on your own computer.** Your computer must join the
  `192.168.123.x` network. Two physical options:
  - **Ethernet cable (recommended):** plug your computer into the robot's
    network port and give your NIC a static `192.168.123.x` address (e.g.
    `192.168.123.222/24`). Then `g1_deploy_onnx_ref` can DDS-talk to the robot
    exactly as if it were on-board. Most reliable; required if you want the
    real-time control loop off-board.
  - **Wi-Fi:** if the robot is bridged onto a Wi-Fi network (see
    `gear_sonic_deploy/scripts/setup_wifi.sh`, which configures the robot's Wi-Fi)
    and your computer is on the same network *with a route to `192.168.123.x`*,
    DDS can work — but Wi-Fi adds latency and jitter to a 500 Hz control loop and
    is **not recommended for the body control loop**. Prefer Wi-Fi only for the
    non-realtime ZMQ links (teleop/camera/exporter), not for `rt/lowcmd`.

  When running off-board, pass the interface explicitly if auto-detect picks the
  wrong one: `./deploy.sh enp0s31f6` or `./deploy.sh 192.168.123.222`.

**(b) The ZMQ links — plain TCP, can cross any reachable network.**
Everything *between* the Python helpers and the deploy binary is ZMQ over TCP:
manager→deploy (`command`/`planner`, 5556), deploy→manager/exporter feedback
(`g1_debug`, 5557), camera server→viewers/exporter (5555). These do **not** need
to be on `192.168.123.x` — they only need IP reachability between the two
endpoints, and each connecting side takes a `--*-host` flag:

- If a component runs on the **same machine** as the deploy binary, leave the
  host at `localhost`.
- If a component runs on a **different machine**, set its `--zmq-host` /
  `--camera-host` / `--relay-host` / `--feedback-host` to the **IP of the
  machine that binds that port**. Make sure the port is reachable (no firewall
  blocking 5555/5556/5557/5559/10000).

> **Who binds vs who connects** matters for ZMQ. The **manager binds** 5556 and
> the **deploy connects** to it; the **deploy binds** 5557 (`g1_debug`) and the
> **manager/exporter connect**; the **camera server binds** 5555 and the
> **viewer/exporter connect**. So the "host" you pass is always the IP of the
> *binding* side. See the ports table in §10.

**Three common topologies:**

| Topology | Body DDS | ZMQ hosts |
|---|---|---|
| All on-board (simplest) | Jetson on `192.168.123.x` | everything `localhost` |
| Deploy on-board, teleop/exporter on your PC | Jetson | manager/exporter use `--feedback-host <jetson-ip>`; deploy uses `--zmq-host <your-pc-ip>`; camera server on Jetson, viewer/exporter use `--camera-host <jetson-ip>` |
| Deploy on your PC (cabled to robot) | your PC on `192.168.123.x` | run the rest on your PC too → `localhost` |

---

## 2. What changed in this fork (since `3e8d73f`)

For orientation, the deployment-relevant deltas are:

**Control stack (`gear_sonic_deploy/`)**
- `src/g1/g1_deploy_onnx_ref/include/hand_config.hpp` — **new**. Compile-time
  hand selector (`USE_BRAINCO_HANDS`, `NUM_HAND_MOTORS`).
- `src/g1/g1_deploy_onnx_ref/include/brainco_hands.hpp` — **new**. DDS-client
  driver for the two BrainCo hands, drop-in compatible with `dex3_hands.hpp`.
- `src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` — switched from a
  hard-coded `Dex3Hands` to a `HandDriver` typedef selected by `hand_config.hpp`;
  generalized all `7`→`NUM_HAND_MOTORS`; added a BrainCo keyboard-fallback
  close/open path; reads `states()[i].q()` (BrainCo) vs `motor_state()[i].q()`
  (Dex3).
- `include/input_interface/keyboard_handler.hpp` — `x`/`c` keys now adjust hand
  close ratio (±0.1).
- `include/state_logger.hpp` — hand CSV columns documented as 7 (Dex3) / 6
  (BrainCo).
- `thirdparty/brainco_hand_service/` — **vendored** Unitree serial→DDS bridge
  (cloned from `unitreerobotics/brainco_hand_service`; currently **untracked**
  in this fork — see §3.6).
- `docker/quest_relay/` — **new** Quest→ZMQ relay container + launcher.

**Teleop / sim side (`gear_sonic/`)**
- `scripts/quest_manager_thread_server.py` — **new** (1300+ lines) Quest teleop
  manager (documented in `docs/quest_manager_docs.md`).
- `data/.../model_data/g1/` reorganized into `with_dex3/` and `with_brainco/`
  (this is the "moved default g1 to hand-specific folder" commit).
- `utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_brainco.yaml` — **sim**
  config for BrainCo (only relevant to MuJoCo, not real-robot deploy).
- `install_scripts/install_pico.sh` — now also installs the
  `third_party/brainco-retargeting[live]` package into `.venv_teleop`.
- Submodules: `third_party/{brainco-retargeting, ROS-TCP-Endpoint, FoundationPose}`.

> Pose-estimation / FoundationPose work (later commits) is orthogonal to
> teleop deployment and is not covered here.

---

## 3. PART A — BrainCo hands (the part to get right first)

The BrainCo integration is split across **two pieces that must agree**:

1. `brainco_hand_service` — a standalone C++ process that owns the physical
   serial connection and exposes the hands on DDS. **Not built by the deploy
   CMake; you build and run it separately.**
2. `BraincoHands` (in `brainco_hands.hpp`) — the DDS *client* compiled into
   `g1_deploy_onnx_ref`, enabled by `USE_BRAINCO_HANDS=1`.

If these disagree (service not running, wrong topics, wrong motor count) the
deploy binary's arms will still move but the fingers will be dead.

### 3.1 Hardware & wiring

- BrainCo Revo2 = **6 actuated DOF per hand**, controlled over **RS485/Modbus**
  via a **USB-to-serial** adapter (one per hand). On the Jetson they enumerate
  as `/dev/ttyUSB*` (the service also accepts `/dev/ttyHAND*`, `/dev/ttyUN*`).
- Each hand has a fixed **Modbus slave ID**: **left = `0x7e`, right = `0x7f`**;
  **baud = 460 800** (`thirdparty/brainco_hand_service/main.cpp:22-24`).
- The service **auto-discovers** which physical port is which hand by reading
  the device SKU (`SKU_TYPE_*_LEFT` / `_RIGHT`) — you do **not** need to pin
  `/dev/ttyUSB0` vs `ttyUSB1` to a side. But both adapters must be plugged in
  and powered before the service starts.
- The service process needs serial access: run as a user in the **`dialout`**
  group (the autostart unit sets `Group=dialout`) or via `sudo`.

### 3.2 Value convention (memorize this)

Everything BrainCo is **normalized to `[0.0, 1.0]`**:

- position `q`: **0.0 = fully open**, **1.0 = fully closed**
- speed `dq`: 0.0 = stopped, **1.0 = full speed (recommended default)**
- 6 fingers, order **`[Thumb, Thumb_aux, Index, Middle, Ring, Pinky]`**

The serial bridge maps `[0,1] → [0,1000]` integer counts for the Stark SDK
(`main.cpp:106-107`) and reports `currents` as `tau_est` for monitoring.

The retargeter's wire order (from the Quest manager) is
`[thumb_metacarpal, thumb_proximal, index_proximal, middle_proximal,
ring_proximal, pinky_proximal]` + a `0.0` pad (7th slot, ignored) — this matches
the firmware's 6-motor layout. See `docs/quest_manager_docs.md` §7.

### 3.3 DDS topic contract

| Direction | Left | Right |
|---|---|---|
| Command (`MotorCmds_`) | `rt/brainco/left/cmd` | `rt/brainco/right/cmd` |
| State (`MotorStates_`) | `rt/brainco/left/state` | `rt/brainco/right/state` |

- The **service** subscribes `*/cmd` and publishes `*/state` at **100 Hz**
  (`main.cpp:146`).
- The **deploy driver** (`BraincoHands`) publishes `*/cmd` and subscribes
  `*/state`, at the deploy command cadence (~500 Hz). It reads
  `states()[i].q()`/`.dq()` (note: BrainCo uses `states()`, Dex3 uses
  `motor_state()` — handled by the `#if USE_BRAINCO_HANDS` branches in the cpp).

Both sides reuse the message types `unitree_go::msg::dds_::MotorCmds_` /
`MotorStates_` sized to **6** elements.

### 3.4 Building `brainco_hand_service` on the Jetson

It has its **own** CMake build, independent of the deploy stack. From the
vendored copy:

```bash
# On the Jetson (aarch64). unitree_sdk2 must be installed system-wide first.
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2
cd unitree_sdk2 && mkdir build && cd build && cmake .. && sudo make install

# Build the service
sudo apt install libspdlog-dev libfmt-dev libboost-program-options-dev libyaml-cpp-dev
cd <repo>/gear_sonic_deploy/thirdparty/brainco_hand_service
mkdir build && cd build
cmake .. && make -j6
# Binaries land in ../bin/{brainco_hand_server, test_brainco_hand_server}
```

CMake details worth knowing (`thirdparty/brainco_hand_service/CMakeLists.txt`):
- It **hard-fails** on any architecture other than `aarch64`/`x86_64`.
- It links the vendored BrainCo SDK
  `lib/${CMAKE_SYSTEM_PROCESSOR}/libbc_stark_sdk.so` (both `aarch64` and
  `x86_64` `.so` files are present in `lib/`), plus `unitree_sdk2 ddsc ddscxx
  yaml-cpp fmt boost_program_options`.
- Headers `include/stark-sdk.h`, `param.h`, `dds/*.h` are vendored.

> There is also `thirdparty/roboticsservice_1.0.0.0_arm64.deb` — the Unitree
> robotics service package (`arm64`). Install with `sudo dpkg -i` if the robot's
> base service stack is not already present. This is the same package Unitree
> ships; only needed if the on-board image lacks it.

### 3.5 Testing the hands (before touching the control stack)

```bash
cd <repo>/gear_sonic_deploy/thirdparty/brainco_hand_service/bin
# Start the bridge (defaults to a sensible interface; or pass --network eth0)
sudo ./brainco_hand_server --network <robot_iface>
# In another terminal — fingers should repeatedly fist + open:
sudo ./test_brainco_hand_server left
sudo ./test_brainco_hand_server right
```

If the hands cycle open/closed, the serial link, IDs, baud, and DDS are all
good. **Do not proceed to the control stack until this works.**

### 3.6 Auto-start as a service (recommended for real runs)

```bash
cd <repo>/gear_sonic_deploy/thirdparty/brainco_hand_service
bash setup_autostart.sh
```

This installs `/etc/systemd/system/brainco_hand.service`:
- `User=unitree`, `Group=dialout`, `Restart=always`, `RestartSec=5`
- `ExecStart=<.../bin>/brainco_hand_server`
- `LD_LIBRARY_PATH` points at `lib/$(uname -m)` so `libbc_stark_sdk.so` resolves.

Manage it with:
```bash
sudo systemctl status  brainco_hand.service
sudo journalctl -u     brainco_hand.service -f
sudo systemctl restart brainco_hand.service
```

> **Important caveat for this fork:** `gear_sonic_deploy/thirdparty/brainco_hand_service/`
> is currently **untracked** (`git status` shows `?? .../brainco_hand_service/`).
> It is a local clone, not yet committed or wired up as a submodule. Make sure
> it is actually present on the robot you deploy to — a fresh `git clone` of this
> fork will **not** bring it down. Either commit it / add it as a submodule, or
> re-clone it on the target (`git clone
> https://github.com/unitreerobotics/brainco_hand_service`). The autostart unit
> uses an absolute path to wherever you put it, so the location must be stable.

### 3.7 How the deploy binary drives the hands

Inside `g1_deploy_onnx_ref.cpp` (all selected by `hand_config.hpp`):

- `using HandDriver = BraincoHands;` when `USE_BRAINCO_HANDS=1`.
- `hand_.initialize("")` — reuses the already-initialized `ChannelFactory`
  (the body init calls `ChannelFactory::Init(0, iface)` first).
- During the init ramp, hands are commanded **open**; on close-out, **closed**
  then damping.
- Every publish tick: `hand_.setAllJointsCommand(side, buffer)` then
  `hand_.writeOnce()`.
- **Delta-q smoothing** (`brainco_hands.hpp:124-156`): `writeOnce()` clamps each
  motor's per-tick change to `MAX_DELTA_Q = 0.05` *relative to the measured
  state*. At ~500 Hz this caps closing speed and prevents jumps when the
  streamed command changes abruptly. This is why state feedback from the service
  matters even though we command positions — without `*/state`, smoothing falls
  back to clamping desired-vs-desired only.
- `setAllJointsCommand` accepts a **7-element** array (call-site compatibility
  with Dex3's 7-DOF buffer / the manager's 7-slot wire vector) but only uses the
  first **6** elements.
- `SetMaxCloseRatio()` / `GetMaxCloseRatio()` are **no-ops** for BrainCo (it
  uses normalized values directly), so the `HandCloseRatio` print will read
  `1.0`.

### 3.8 Keyboard fallback (no Quest)

When there is **no external hand data** (e.g. `--input-type keyboard`, or the
manager not yet streaming), the cpp synthesizes a uniform finger command from
the keyboard close-ratio (`#if USE_BRAINCO_HANDS` block around
`g1_deploy_onnx_ref.cpp:2995`):

- `x` = close more, `c` = open more (the keyboard handler adjusts the ratio by
  ±0.1).
- The ratio range `[0.2, 1.0]` is linearly mapped to BrainCo `[0.0 open … 1.0
  closed]`. Initial value `0.2` ⇒ hands start fully open.

This lets you sanity-check the hands through the full control stack without the
Quest pipeline.

### 3.9 BrainCo troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Arms move, fingers dead | `brainco_hand_service` not running | `systemctl status brainco_hand`; start it |
| Service: "No ttyUSB serial ports found" | adapters unplugged / no power / wrong `/dev` | check `ls /dev/ttyUSB*`, USB cabling, hand power |
| Service: "Port … is not left/right hand (sku …)" | wrong SKU / firmware | confirm hands are Revo2 SMALL/MEDIUM L/R variants |
| Permission denied on `/dev/ttyUSB*` | not in `dialout` | add user to `dialout` or run with `sudo` |
| Fingers jump / overshoot | smoothing off (no state) or stream glitch | verify `*/state` is publishing; check DDS iface match |
| `libbc_stark_sdk.so` not found at runtime | `LD_LIBRARY_PATH` | point at `lib/$(uname -m)` (autostart unit does this) |
| Deploy ignores hands entirely | binary built with `USE_BRAINCO_HANDS=0` | set to `1` in `hand_config.hpp`, **rebuild** |

---

## 4. PART B — Deploying the control stack (`g1_deploy_onnx_ref`)

### 4.1 On-board dependencies

The deploy binary is a native CUDA/TensorRT program. On the Jetson it needs:

- **CUDA toolkit** (CMake auto-detects `/usr/local/cuda*`; ARM uses
  `targets/aarch64-linux/lib`).
- **TensorRT** — set `export TensorRT_ROOT=$HOME/TensorRT` in `~/.bashrc`
  (deploy.sh warns if unset; download from
  developer.nvidia.com/tensorrt/download/10x).
- **ONNX Runtime** — `setup_env.sh` searches `/opt/onnxruntime`, etc., and sets
  `onnxruntime_DIR`. If missing, run `scripts/install_deps.sh`.
- **libzmq** (`libzmq3-dev`), **msgpack** (`libmsgpack-dev`), **Eigen3**,
  **ZLIB**.
- **`just`**, **CMake**, **clang/gcc**, **git** (deploy.sh runs
  `scripts/install_deps.sh` if `just`/tools are missing).
- **DLA / cudla**: on non-Thor Jetson the CMake links `cudla` for DLA offload;
  on Thor it is skipped automatically. Detected from `/proc/device-tree/model`
  or `IS_THOR=true`.
- **ROS2 (optional)**: `setup_env.sh` auto-sources the newest of
  `jazzy/iron/humble/...` and sets `HAS_ROS2=1`,
  `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_LOCALHOST_ONLY=1`. The CMake gates
  the ROS2 input/output handlers behind `HAS_ROS2`. ROS2 is **only needed if you
  use `--output-type ros2/all` or `--input-type ros2`**. For the Quest pipeline
  you can run pure ZMQ and skip ROS2 (`HAS_ROS2=0`).

`scripts/install_deps.sh`, `scripts/install_ros2_humble.sh`, and
`scripts/setup_env.sh` automate most of this. `deploy.sh` sources `setup_env.sh`
for you.

### 4.2 Model artifacts (must be present)

The defaults in `deploy.sh` point at:

| Flag | Default path | What it is |
|---|---|---|
| checkpoint (decoder) | `policy/release/model_decoder.onnx` | SONIC policy decoder |
| checkpoint (encoder) | `policy/release/model_encoder.onnx` | observation encoder |
| obs config | `policy/release/observation_config.yaml` | obs layout for the policy |
| planner | `planner/target_vel/V2/planner_sonic.onnx` | locomotion planner |
| motion data | `reference/example/` | reference motions (key-frame library) |

Prebuilt TensorRT engines (`*.trt`) are also present
(`policy/release/policy_model_decoder.trt`, `encoder_model_encoder.trt`,
`planner/.../planner_planner_sonic.trt`) but TRT engines are **hardware/driver
specific** — if TensorRT/GPU differs from where they were built, the binary will
rebuild engines from the ONNX on first run (slower startup, expected). Make sure
the ONNX files exist. Pull LFS first: `git lfs pull`.

> These model files are **independent of hand type** — the policy controls the
> body, not the fingers. You do not need a "BrainCo policy".

### 4.3 Build with the BrainCo switch

The hand type is **compile-time**. Verify before building:

```bash
grep USE_BRAINCO_HANDS gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/hand_config.hpp
# → #define USE_BRAINCO_HANDS 1     (this fork ships it as 1)
```

If you ever flip it, you **must rebuild** (`just clean && just build`, or just
re-run `deploy.sh` which always rebuilds). The build:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh      # sets TensorRT/onnx/ROS2/CMake paths
just build                       # cmake -S .. -B build && cmake --build
# Binary: gear_sonic_deploy/target/release/g1_deploy_onnx_ref
```

`deploy.sh` does all of the above (deps check → `setup_env.sh` → `just build`)
and then launches.

### 4.4 Launching with `deploy.sh`

`deploy.sh` resolves the network interface, checks model files, builds, and runs
the binary. Two modes:

```bash
cd gear_sonic_deploy

# REAL robot — auto-detects the 192.168.123.x interface
./deploy.sh real

# SIMULATION (MuJoCo on loopback) — adds --disable-crc-check
./deploy.sh sim

# Explicit interface or IP also accepted:
./deploy.sh enP8p1s0
./deploy.sh 192.168.123.164
```

What `real` resolves to (`deploy.sh:148-171`): it scans for an interface holding
a `192.168.123.*` address (the Unitree robot LAN). If none is found it falls
back to the first non-loopback iface (with a warning), or finally `enP8p1s0`.
**Confirm the printed "Resolved interface" is actually your robot link** — a
wrong interface means the binary talks DDS into the void.

The exact command it runs (defaults, see `deploy.sh:238-244, 556-572`):

```bash
just run g1_deploy_onnx_ref <iface> policy/release/model_decoder.onnx reference/example/ \
    --obs-config    policy/release/observation_config.yaml \
    --encoder-file  policy/release/model_encoder.onnx \
    --planner-file  planner/target_vel/V2/planner_sonic.onnx \
    --input-type    manager \
    --output-type   all \
    --zmq-host      localhost
# (sim adds: --disable-crc-check)
```

Override any of these via flags, e.g.
`./deploy.sh --zmq-host 192.168.123.10 --output-type zmq real` if the Quest
manager runs off-board and you don't need ROS2.

**Running on the robot vs. on your computer (recap from §1):**

- **On the robot (Way 1):** just `./deploy.sh real` on the Jetson. The interface
  auto-resolves to the internal `192.168.123.x` link. Keep `--zmq-host localhost`
  if the manager/exporter also run on the Jetson; otherwise point it at the PC
  running the manager.
- **On your computer (Way 2):** first put your computer on `192.168.123.x` (cable
  recommended, §1). Then `./deploy.sh real` (or pass the interface/IP explicitly
  if auto-detect is wrong: `./deploy.sh 192.168.123.222`). Set `--zmq-host` to
  wherever the manager runs (`localhost` if also on your PC). The control loop
  now runs on your PC and commands the robot over the cable — make sure the link
  is solid; a flaky cable/Wi-Fi here directly degrades the 500 Hz control loop.

> Note: `deploy.sh` currently runs **non-interactively** — the `read -p`
> confirmation prompt is commented out. Running `./deploy.sh real` will
> **immediately** build and start commanding the real robot. Make sure the robot
> is suspended/safe and an e-stop is in reach before you run it.

### 4.5 Input / output types

`--input-type` options (`g1_deploy_onnx_ref.cpp:2455-2514`):

| Value | Meaning |
|---|---|
| `manager` | **Default for Quest.** `InterfaceManager` — ZMQ-driven, runtime-switchable between keyboard/gamepad/zmq via Shift+1..4. Subscribes to the manager's `command`/`planner` topics. |
| `zmq_manager` | Direct ZMQ manager (pose + `command` + `planner` topics). |
| `zmq` | Raw ZMQ pose endpoint. |
| `keyboard` / `gamepad` / `gamepad_manager` | Local input (use `keyboard` for hand-only fallback tests, §3.8). |
| `ros2` | ROS2 input (needs `HAS_ROS2=1`). |

`--output-type` (`...cpp:2562-2588`): `zmq` (binds `g1_debug` PUB on **5557**),
`ros2`, or `all` (both). The Quest manager subscribes to the ZMQ `g1_debug`
feedback at 5557 to read measured joints for recalibration (`r` key), so use
`all` or `zmq`.

ZMQ host/port: `--zmq-host` (default `localhost`), `--zmq-port` (default
**5556**, where the manager publishes), `--zmq-out-port` (default **5557**).

### 4.6 Runtime sequence, state machine, controls

On start the binary:
1. Inits DDS on the chosen interface; inits the hand driver.
2. Ramps to the default pose (`ProgramState`: init ramp → `WAIT_FOR_CONTROL`),
   hands open.
3. Waits for a **START** command (sent by the manager when you press `s`) before
   it leaves `WAIT_FOR_CONTROL` and begins tracking.
4. Streams body torque commands at the control rate and republishes hand
   commands every publish tick.

Keyboard controls while running (BrainCo variant):
- `x` / `c` — close / open all fingers (steps ~0.125 normalized) **when no Quest
  data is present**.
- `g`/`h`, `b`/`v` — legacy Dex3 per-hand close-ratio nudges (no-op effect on
  BrainCo finger positions, but still print).
- The status line prints `HandCloseRatio` (always `1.0` for BrainCo).

`--disable-crc-check` is **sim only**; never pass it on the real robot (CRC
validates the Unitree low-state packets).

### 4.7 Logging

`StateLogger` writes per-tick CSVs (joint `q`/`dq`, `action`, motor temps,
`left/right_hand_q|dq`, `*_hand_action`, encoder token/mode). For BrainCo the
hand CSVs have **6 columns** (vs 7 for Dex3) — see `state_logger.hpp`. Logs land
under `gear_sonic_deploy/logs/`. Useful for verifying the fingers actually
received and tracked commands.

### 4.8 Control-stack troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Missing file" at startup | LFS not pulled / wrong checkpoint path | `git lfs pull`; check `policy/release/*` |
| Robot never leaves `WAIT_FOR_CONTROL` | no START / manager not connected | check ZMQ host/port; press `s` in manager |
| Talks DDS but robot doesn't respond | wrong interface | verify "Resolved interface" = 192.168.123.x link |
| Slow first start | rebuilding TRT engines from ONNX | expected on new HW; subsequent runs cached |
| `TensorRT_ROOT not set` | env | `export TensorRT_ROOT=$HOME/TensorRT` |
| ROS2 handler missing | `HAS_ROS2=0` | source ROS2 / install humble, or use ZMQ only |
| CRC errors on real robot | `--disable-crc-check` left on | remove it (it's sim-only) |

---

## 5. PART C — Quest teleop (turn-key; brief)

The Quest path is well-trodden; full detail is in
**`docs/quest_manager_docs.md`**. Minimum to bring it up:

### 5.1 Relay container

```bash
# From repo root (build context must be repo root so it can COPY submodules)
python gear_sonic_deploy/docker/quest_relay/run_quest_relay.py
# (--rebuild to force a rebuild, --detach to background)
```

This builds & runs a `ros:noetic` container that runs `roscore` +
`ros_tcp_endpoint` (Quest connects over Unity ROS-TCP-Connector on **TCP
10000**) + `relay.py`, which republishes everything as a single msgpack
`quest_data` blob on **ZMQ 5559**. It vendors the `vr_haptic_msgs` ROS1
messages and uses `endpoint_no_adb.launch` (Quest connects over Wi-Fi/TCP, no
adb). Requires `third_party/ROS-TCP-Endpoint` (submodule).

On the Quest: install/launch the Unity teleop app and point its ROS-TCP
connector at the host running the relay, port 10000.

### 5.2 Manager

```bash
source .venv_teleop/bin/activate        # created by install_scripts/install_pico.sh
python gear_sonic/scripts/quest_manager_thread_server.py \
    --relay-host <relay_host> --relay-port 5559 \
    --port 5556 \
    --feedback-host <deploy_host> --feedback-port 5557
```

It publishes `command`/`planner`/`manager_state` on **5556** (the deploy
binary's `--zmq-host:--zmq-port`) and subscribes to the deploy's `g1_debug`
feedback on **5557**. The `install_pico.sh` change pulls in
`third_party/brainco-retargeting[live]`, which provides the optimization-based
**MANO-21 → 6 BrainCo motor** retargeter (with a pure-numpy fallback via
`--np-retarget`). Finger output is the 7-slot wire vector consumed by the
deploy hand driver.

### 5.3 Operating

1. Operator stands in the **rest pose** mirroring the robot.
2. Press **`s`** → 3 s countdown → calibration → policy START → VR-3pt tracking.
3. **`r`** recalibrate (uses measured robot joints), **`p`** pause/resume,
   **`f`** toggle fingers, **`c`/`x`** data collection toggles, **`q`** stop.
4. Walking is driven by head planar velocity; `--disable-walk` (turn-in-place
   only) or `--static-base` (arms/hands only) to restrict motion while testing.

### 5.4 Testing teleop without a headset

- **Replay**: `--replay path/to/traj.npz` plays a recorded trajectory with a
  synthetic rest-pose prefix (calibrates on frame 0).
- **Mock streamer**: `gear_sonic/scripts/mock_quest_streamer.py` /
  `docker/quest_relay/generate_mock_quest_data.py` synthesize `quest_data` so you
  can exercise the full manager→deploy→robot/hands chain with no Quest.

This is the recommended way to validate the **control stack + BrainCo hands**
end-to-end before involving the headset.

---

## 6. PART D — Cameras & data collection

The end goal of the project is to **record manipulation trajectories** (robot
proprioception + teleop targets + camera video, and optionally object pose).
This is done by three cooperating Python processes that sit *alongside* the
control stack and consume the same ZMQ streams it already publishes. None of
them touch the robot directly — they are all read-only subscribers — so they are
safe to start/stop independently while the robot is running.

```
  cameras (OAK/RealSense/USB)
        │
        ▼
  camera server  ──ZMQ 5555──▶  camera viewer   (see what the robot sees)
  (composed_camera)      │
                         └────▶  data exporter   ──▶  LeRobot dataset on disk
                                   ▲   ▲   ▲
            g1_debug (5557) ───────┘   │   │
            pose/planner/manager_state │   │
                          (5556) ──────┘   │
            robot_config (5557) ───────────┘
```

### 6.1 The camera server — "what the robot sees"

`gear_sonic.camera.composed_camera` is the camera server. It runs **on the
machine the cameras are physically plugged into** (normally the robot's Jetson),
drives one or more cameras (each in its own thread, with auto-reconnect), and
publishes all frames merged into a single ZMQ `ImageMessageSchema` payload on
**port 5555**.

```bash
# On the robot (where the cameras are connected), in .venv_data_collection:
source .venv_data_collection/bin/activate
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera oak \
    --ego-view-device-id <OAK_SERIAL> \
    --port 5555
# Supported camera types: oak, oak_mono, realsense, usb, or a path to an
# .mp4 file for replay testing. Run with --help for the full option list.
```

Drivers available: `gear_sonic/camera/drivers/{oak,realsense,usb_camera,dummy}.py`.
Use `dummy`/an `.mp4` path to develop the pipeline with no hardware.

### 6.2 Seeing the feed live — the camera viewer

`run_camera_viewer.py` is a ROS-free OpenCV viewer that connects to the camera
server and displays the live tiles. This is the direct "see what the robot sees"
tool.

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host <camera-server-ip> --camera-port 5555
# Controls (window focused): R = start/stop MP4 recording, Q = quit
# Recordings land in camera_recordings/rec_<timestamp>/<stream>.mp4
```

- **Camera server and viewer on the same machine** (e.g. both on the Jetson):
  `--camera-host localhost`.
- **Viewer on your computer, cameras on the robot:** run the server on the robot
  and set `--camera-host <jetson-ip>` on your computer. 5555 must be reachable
  (this can go over Wi-Fi; it's not the realtime control loop). This is the usual
  way to watch the feed from your desk while the robot operates.

The viewer is independent of recording a dataset — it's just a monitor (with its
own optional MP4 dump). Skip it during a real session with
`launch_data_collection.py --no-camera-viewer` if you don't need it.

### 6.3 Recording a dataset — the data exporter

`run_data_exporter.py` is the recorder. It builds a **LeRobot v2.1 dataset**
(parquet + mp4) by subscribing to everything the rest of the stack already
publishes (`run_data_exporter.py` header + `_handle_*` methods):

| Source | Topic | Port | Bound by | Contributes |
|---|---|---|---|---|
| Robot state | `g1_debug` | 5557 | deploy (`--output-type zmq/all`) | proprio: `body_q`, hand `q`, actions, base quat |
| Robot config | `robot_config` | 5557 | deploy (re-published ~2 s) | `script_config` in `info.json` (startup gate) |
| Teleop pose | `pose` | 5556 | manager | SMPL pose / hand joints |
| Planner | `planner` | 5556 | manager | VR 3-point targets, commands |
| Manager state | `manager_state` | 5556 | manager | **record/abort toggles** + stream mode |
| Cameras | (ZMQ) | 5555 | camera server | RGB(-D) video tiles |

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the box" \
    --camera-host <camera-ip>   --camera-port 5555 \
    --sonic-zmq-host <manager-ip>  --sonic-zmq-port 5556 \
    --state-zmq-host <deploy-ip>   --state-zmq-port 5557 \
    --dataset-name my_session            # optional
# (hosts default to localhost; set them per §1 if pieces run on different boxes)
```

> **Startup gate:** the exporter waits for the `robot_config` ZMQ message from
> the C++ deploy before it will record (`--robot-config-timeout`, `0` = wait
> forever). If it hangs at startup, the deploy binary isn't publishing
> `robot_config` — check `--output-type` is `zmq`/`all` and the
> `--state-zmq-host`/port point at the deploy binary.

### 6.4 Enabling / triggering recording

You do **not** start/stop the exporter per episode — you toggle recording while
it runs. Two equivalent triggers, both consumed by the exporter's
`EpisodeState` (IDLE → RECORDING → NEED_TO_SAVE → IDLE):

1. **From the Quest manager (primary).** In
   `quest_manager_thread_server.py`: press **`c`** to toggle data collection and
   **`x`** to toggle abort. These set `toggle_data_collection` /
   `toggle_data_abort` on the `manager_state` topic (5556), which the exporter
   reads in `_handle_manager_state`. So a single operator at the headset can arm
   recording, run the episode, and save — no second keyboard needed.
2. **From a ZMQ keyboard subscriber (alternative).** The exporter also listens on
   a ZMQ keyboard channel (port **5580**, `ZMQKeyboardSubscriber`) for the same
   toggles, useful for scripted or off-headset control.

Lifecycle of one episode:
- Toggle **on** (`c`) → `RECORDING`: frames accumulate.
- Toggle again (`c`) → `NEED_TO_SAVE`: the episode is flushed to the dataset.
- **Abort** (`x`) → discard the in-progress episode (e.g. a botched grasp)
  without writing it.

Object pose (FoundationPose) is written per-episode via `FoundationPoseWriter`
when enabled — that pipeline is documented separately and is orthogonal to the
core robot+camera recording.

### 6.5 One-command launcher (optional)

`launch_data_collection.py` brings the whole stack up in a single **tmux**
session (deploy + exporter + teleop + camera viewer panes):

```bash
python gear_sonic/scripts/launch_data_collection.py            # real robot
python gear_sonic/scripts/launch_data_collection.py --sim      # MuJoCo sim
python gear_sonic/scripts/launch_data_collection.py --no-camera-viewer
```

> Caveat: this launcher predates the Quest manager and wires up the **PICO**
> teleop pane (`pico_manager_thread_server.py`). For the Quest workflow, either
> start the pieces manually (camera server → deploy → exporter → quest relay →
> quest manager) or adapt the launcher's teleop pane to run
> `quest_manager_thread_server.py`. The deploy/exporter/camera panes are still
> correct. It also needs `.venv_data_collection` and `.venv_teleop`
> (`install_scripts/install_data_collection.sh`, `install_pico.sh`).

### 6.6 Post-processing the recordings

`process_dataset.py` cleans and merges datasets directly on the LeRobot v2.1
on-disk format (no training framework needed):

```bash
# Clean stale/frozen SMPL frames (teleop pauses, ZMQ drops), in place or to a copy
python gear_sonic/scripts/process_dataset.py --dataset-path outputs/my_session
python gear_sonic/scripts/process_dataset.py --dataset-path outputs/my_session \
    --output-path outputs/my_session_cleaned

# Merge sessions (validates matching script_config)
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/s1 outputs/s2 --output-path outputs/merged
```

Other inspection helpers: `visualize_recording.py`,
`visualize_teleop_data_recording.py`,
`visualize_robot_object_trajectory.py` (robot + object scene replay).

### 6.7 Data collection — where to run what (both ways)

The exporter and camera viewer are pure ZMQ subscribers, so they can sit on
**either** machine:

- **Everything on the robot:** camera server, deploy, exporter all on the Jetson;
  every `--*-host` stays `localhost`. Lowest fuss; you'll review the dataset on
  the Jetson or copy it off afterward.
- **Record from your computer:** run the camera server **on the robot** (cameras
  are physically there) and run the **exporter + viewer on your computer**,
  pointing `--camera-host` at the Jetson and `--state-zmq-host` /
  `--sonic-zmq-host` at whichever machines bind 5557 (deploy) and 5556 (manager).
  These ZMQ streams tolerate Wi-Fi; only the **body DDS control link** wants the
  cable (§1). A common split: deploy on the robot (Way 1), teleop + exporter +
  viewer on your PC.

---

## 7. Full bring-up checklist (real robot)

On the robot's on-board compute:

1. `git lfs pull` (models/meshes). Ensure `policy/release/*` and
   `planner/target_vel/V2/*` exist.
2. Ensure `thirdparty/brainco_hand_service/` is present (it is **untracked** —
   re-clone if needed, §3.6). Build it (§3.4). Install + start its systemd
   service (§3.6).
3. `sudo ./test_brainco_hand_server left|right` → confirm both hands cycle (§3.5).
4. `grep USE_BRAINCO_HANDS .../hand_config.hpp` → `1`.
5. `export TensorRT_ROOT=$HOME/TensorRT`; ensure CUDA/onnxruntime present
   (`scripts/install_deps.sh` if not).
6. Robot powered, suspended/safe, **e-stop in hand**. Confirm the
   `192.168.123.x` link is up.
7. `cd gear_sonic_deploy && ./deploy.sh real` → check the resolved interface +
   model files in the banner. It will build then start commanding the robot.
8. Start the relay (§5.1) and manager (§5.2) — on-board or dev PC, ports
   reachable.
9. Operator in rest pose → press `s`. Verify body tracks and (with `f` on)
   fingers follow.
10. **(Optional, for recording)** Start the camera server on the robot (§6.1),
    then a camera viewer to confirm the feed (§6.2), then the data exporter with
    a `--task-prompt` (§6.3). Set `--camera-host`/`--state-zmq-host`/
    `--sonic-zmq-host` per your topology (§1, §6.7).
11. **(Optional, for recording)** Arm/save episodes from the Quest manager:
    `c` = start/stop recording, `x` = abort (§6.4). Post-process with
    `process_dataset.py` (§6.6).
12. To stop: `q` in the manager (sends STOP), then Ctrl-C the deploy binary
    (graceful damping shutdown).

---

## 8. Cross-cutting gotchas

- **Order matters:** BrainCo service up **before** the deploy binary, deploy
  binary up **before** the manager presses START (or the manager will reconnect,
  but cleaner to start in order).
- **`USE_BRAINCO_HANDS` is compile-time.** No runtime flag. Wrong value → wrong
  motor count, wrong DDS state field (`states()` vs `motor_state()`), dead or
  mis-indexed fingers. Rebuild on change.
- **One DDS interface for everything.** Body firmware, `brainco_hand_service`,
  and the deploy binary must share the same network interface / DDS domain
  (domain `0`). If the hand service is launched on a different iface than the
  deploy binary, `*/state` and `*/cmd` won't meet.
- **`brainco_hand_service` is untracked in this fork.** It will not come down
  with a fresh clone. Commit it / submodule it, or document the re-clone step in
  your runbook — otherwise a clean deploy host silently has no hands.
- **`deploy.sh` runs unattended** (confirmation prompt commented out) and
  `--disable-crc-check` must never be set on the real robot.
- **TRT engines are not portable.** Ship the ONNX; let the target rebuild
  engines. Don't rely on the committed `*.trt` matching the robot's
  TensorRT/driver.
- **Hand smoothing depends on state feedback.** `MAX_DELTA_Q=0.05` smoothing
  uses measured `*/state`; if the service isn't publishing state, the fingers
  lose the rate limit relative to actual position.
- **The policy is hand-agnostic** — don't go looking for a separate BrainCo
  policy checkpoint; there isn't one and there shouldn't be.

---

## 9. File / path quick reference

| Path | Role |
|---|---|
| `gear_sonic_deploy/deploy.sh` | Top-level build+launch wrapper |
| `gear_sonic_deploy/.justfile`, `scripts/setup_env.sh`, `scripts/install_deps.sh` | Build env + deps |
| `.../g1_deploy_onnx_ref/include/hand_config.hpp` | **Compile-time hand switch** |
| `.../g1_deploy_onnx_ref/include/brainco_hands.hpp` | BrainCo DDS client driver |
| `.../g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` | Main control loop |
| `gear_sonic_deploy/thirdparty/brainco_hand_service/` | Serial→DDS bridge (build + run separately) |
| `gear_sonic_deploy/thirdparty/roboticsservice_1.0.0.0_arm64.deb` | Unitree base service pkg |
| `gear_sonic_deploy/policy/release/`, `planner/target_vel/V2/` | Policy + planner models |
| `gear_sonic_deploy/docker/quest_relay/` | Quest→ZMQ relay container + launcher |
| `gear_sonic/scripts/quest_manager_thread_server.py` | Quest teleop manager |
| `docs/quest_manager_docs.md` | Full manager documentation |
| `third_party/brainco-retargeting/` | MANO-21 → 6 BrainCo motor retargeter |
| `gear_sonic/camera/composed_camera.py`, `camera/drivers/` | Camera server (ZMQ 5555) + drivers |
| `gear_sonic/scripts/run_camera_viewer.py` | Live "what the robot sees" viewer (+MP4) |
| `gear_sonic/scripts/run_data_exporter.py` | Dataset recorder → LeRobot v2.1 |
| `gear_sonic/utils/data_collection/` | Episode state, ZMQ subscribers, FoundationPose writer |
| `gear_sonic/scripts/launch_data_collection.py` | tmux all-in-one launcher (PICO-era; adapt for Quest) |
| `gear_sonic/scripts/process_dataset.py` | Clean / merge recorded datasets |

---

## 10. Ports summary

| Port | Bound by | Consumed by | Payload |
|---|---|---|---|
| TCP 10000 | relay container (`ros_tcp_endpoint`) | Quest Unity app | ROS-TCP-Connector |
| ZMQ 5559 | relay (`relay.py` PUB) | manager (SUB `quest_data`) | msgpack tracking blob |
| ZMQ 5556 | manager (PUB) | deploy (`--input-type manager`, SUB) + exporter (SUB) | `command`/`planner`/`manager_state`/`pose` |
| ZMQ 5557 | deploy (`--output-type zmq/all` PUB) | manager + exporter (SUB) | measured robot state (`g1_debug`) + `robot_config` |
| ZMQ 5555 | camera server (`composed_camera` PUB) | camera viewer + exporter (SUB) | merged camera frames (RGB-D) |
| ZMQ 5580 | external keyboard publisher | exporter (`ZMQKeyboardSubscriber`) | record/abort key events (alt. trigger) |
| DDS domain 0 | firmware + brainco service + deploy | each other | `rt/lowcmd`/`rt/lowstate`, `rt/brainco/*` |

For each ZMQ row, the **"Bound by"** machine's IP is the `--*-host` value the
**"Consumed by"** side must pass (see §1). Only the DDS row must live on
`192.168.123.x`; every ZMQ row can cross Wi-Fi/LAN.
