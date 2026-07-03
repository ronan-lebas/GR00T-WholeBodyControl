# Deployment Setup — Laptop-cabled-to-Robot Topology

This is a practical setup guide for **your specific layout**:

- You connect the robot to your **laptop with an ethernet cable**, and `ssh` into
  the robot.
- You run **almost everything on the laptop**: the deploy binary
  (`g1_deploy_onnx_ref`), the Quest relay, the Quest manager, and the data
  recorder.
- You run **only two things on the robot** (over your ssh session): the
  **camera server** and the **BrainCo hand service**.

For the full conceptual reference (architecture, ports, troubleshooting), see
`deployment_report.md`. This file is the "just tell me what to type" companion,
plus the one piece of networking you need to set up by hand: **giving your
laptop an address on the robot's network (the "NIC" thing).**

---

## 0. Your topology at a glance

```
        LAPTOP (you run these)                         ROBOT  (you ssh in,
   ┌───────────────────────────────┐                   run these)
   │ quest_relay (Docker)          │              ┌─────────────────────────┐
   │ quest_manager_thread_server   │   ETHERNET   │ brainco_hand_service     │
   │ g1_deploy_onnx_ref  (deploy)  │◀════CABLE═══▶│ camera server            │
   │ run_data_exporter (recorder)  │   192.168.   │ (G1 body firmware + IMU) │
   │ run_camera_viewer (optional)  │    123.x     │                          │
   └───────────────────────────────┘              └─────────────────────────┘
```

Two kinds of traffic run over that **one ethernet cable** (SSH is just a third,
unrelated thing also riding the cable — it does **not** carry your robot data):

1. **DDS** (the real-time robot control link): laptop's deploy binary ⇄ the G1
   body motors/IMU **and** ⇄ the BrainCo hand service on the robot. This is the
   link that *requires* your laptop to have an address on `192.168.123.x` — see
   Part 1.
2. **ZMQ port 5555** (camera frames): the camera server on the robot → the
   recorder/viewer on the laptop. Plain TCP; just needs the port reachable.

Everything else (manager ⇄ deploy ⇄ recorder on ports 5556/5557, relay on
5559) is **all on the laptop = `localhost`**, so it never crosses the cable at
all.

---

## Part 1 — The "NIC" thing (giving your laptop an address on the robot network)

### What a NIC is

**NIC = Network Interface Controller** — it's just the technical name for a
network port on your computer. Your laptop's **ethernet jack** is one NIC; your
**Wi-Fi card** is another NIC. Each NIC has a name (like `eth0`, `enp0s31f6`,
`enx00e04c…`) and can be given an **IP address**.

### Why you have to set one up by hand

The robot's internals (body motors, IMU, and the BrainCo hand service) talk on a
private little network with the address range **`192.168.123.x`** (where `x` is
some number 1–254). Normally your laptop's ethernet port gets its address
automatically from a router (DHCP). But the robot is **not a router** — plug a
cable straight into it and nobody hands your laptop an address, so it sits there
with no address on `192.168.123.x` and **cannot talk to the robot**.

So you have to **manually tell your laptop's ethernet NIC: "use the fixed
address `192.168.123.222`."** That's it. That's the whole "NIC thing." Once your
laptop has a `192.168.123.x` address, the deploy binary's DDS can reach the robot
exactly as if it were running on-board.

> We pick **`.222`** for the laptop only because it's almost certainly unused.
> Any free number on `192.168.123.x` works **except** the robot's own (commonly
> `.161`/`.164`) and the firmware/motor addresses.

### Step 1.1 — Find your ethernet NIC's name

Plug the cable into the robot, then run on the **laptop**:

```bash
ip -br link
```

You'll see lines like:

```
lo               UNKNOWN  ...
enp0s31f6        UP       e4:5f:01:...      ← wired ethernet (this is the one)
wlp3s0           UP       ...               ← Wi-Fi (ignore)
```

The wired one usually starts with `en`, `eth`, or `enx`. If you're unsure,
unplug the cable, run it again, and see which line flips to `DOWN`/disappears —
that's your ethernet NIC. **Note the name** (example below uses `enp0s31f6` —
substitute yours).

### Step 1.2 — Give that NIC a static `192.168.123.x` address

Pick **one** of these. Method A is the quick "just for now" version; Method B
survives reboots/replugs.

**Method A — temporary (gone after reboot/unplug, simplest):**

```bash
# Replace enp0s31f6 with YOUR interface name from step 1.1
sudo ip addr add 192.168.123.222/24 dev enp0s31f6
sudo ip link set enp0s31f6 up
```

**Method B — persistent (NetworkManager, the usual Ubuntu default):**

```bash
# Creates a saved profile named "robot" bound to your NIC
sudo nmcli connection add type ethernet ifname enp0s31f6 con-name robot \
    ipv4.method manual ipv4.addresses 192.168.123.222/24
sudo nmcli connection up robot
```

To remove it later: `sudo nmcli connection down robot`.

> The `/24` means "this is a 192.168.123.* network." Don't set a gateway — you
> don't want your normal internet routed through the robot. With Method B above,
> your Wi-Fi keeps providing internet; this profile only adds the robot subnet.

### Step 1.3 — Verify it worked

```bash
ip -br addr show enp0s31f6        # should now list 192.168.123.222/24
ping -c3 <ROBOT_IP>               # e.g. ping -c3 192.168.123.164
```

`<ROBOT_IP>` is **the address you ssh to**. If ping replies, your laptop is on
the robot network and DDS will work. If ping fails: re-check the NIC name, that
the cable is in the robot's network port, and that the robot is powered.

> **Tip:** From now on, the robot's address is whatever you `ssh` to. Find it
> any time with `ip -br addr` *on the robot* (look for the `192.168.123.x`
> line). The examples below call it `<ROBOT_IP>`.

---

## Part 2 (Option A) — Deploy on the LAPTOP: all the commands, with hosts filled in

Open **two kinds of terminals**:

- **[ROBOT]** = a terminal `ssh`'d into the robot.
- **[LAPTOP]** = a local terminal on your laptop.

Set these once so you can copy-paste. On the **laptop**:

```bash
export ROBOT_IP=192.168.123.164        # whatever you ssh to — verify with ip -br addr on the robot
```

> Reminder on the hosts: a `--*-host` flag is always **the IP of the machine
> that *binds* (opens) that port**. The camera server binds 5555 **on the
> robot**, so its consumers use `--camera-host $ROBOT_IP`. Everything else you
> run is on the laptop binding on the laptop, so those hosts stay `localhost`.

### 2.0 — Safety first

Robot powered but **suspended/safe**, **e-stop in your hand**. `deploy.sh` runs
unattended and starts commanding the robot immediately.

### 2.1 — [ROBOT] BrainCo hand service (must be up first)

DDS bridge for the hands. The `-n <iface>` flag is the robot's interface on the
`192.168.123.x` subnet (the same one your laptop is cabled to) so the laptop's
deploy binary can see its `rt/brainco/*` topics. Find the robot's iface name
with `ip -br addr` on the robot (the line holding the `192.168.123.x` address).

```bash
# [ROBOT] — if you installed the systemd service (recommended), just ensure it's up:
sudo systemctl status brainco_hand.service          # green/active = good
# (start/restart if needed)
sudo systemctl restart brainco_hand.service

# --- OR run it by hand instead of the service: ---
cd <repo>/gear_sonic_deploy/thirdparty/brainco_hand_service/bin
sudo ./brainco_hand_server -n <robot_iface>         # e.g. -n eth0
```

Sanity-check the hands cycle open/closed before going further:

```bash
# [ROBOT]
sudo ./test_brainco_hand_server left
sudo ./test_brainco_hand_server right
```

### 2.2 — [ROBOT] Camera server (binds ZMQ 5555)

Runs where the cameras are physically plugged in (the robot).

```bash
# [ROBOT]
source .venv_data_collection/bin/activate
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera oak \
    --ego-view-device-id <OAK_SERIAL> \
    --port 5555
```

(Use `--ego-view-camera realsense`/`usb`, or an `.mp4` path, as appropriate.
Run with `--help` for the full camera options.)

### 2.3 — [LAPTOP] Deploy binary (DDS to robot + binds 5557, subscribes 5556)

The manager runs on the laptop too, so `--zmq-host localhost`. `--output-type
all` (or `zmq`) makes it publish the `g1_debug`/`robot_config` feedback the
manager and recorder need.

```bash
# [LAPTOP]
cd gear_sonic_deploy
./deploy.sh real --zmq-host localhost --output-type all
```

`real` auto-detects your `192.168.123.x` NIC (the one you set up in Part 1).
**Check the printed "Resolved interface"** — it must be your cabled ethernet
port. If it picks the wrong one, name it explicitly:

```bash
# [LAPTOP] — pass your interface name or your laptop's robot-subnet IP
./deploy.sh enp0s31f6 --zmq-host localhost --output-type all
./deploy.sh 192.168.123.222 --zmq-host localhost --output-type all
```

Wait until it reaches `WAIT_FOR_CONTROL` (ramped to default pose) before
continuing.

### 2.4 — [LAPTOP] Quest relay (binds TCP 10000 + ZMQ 5559)

```bash
# [LAPTOP] — from repo root
python gear_sonic_deploy/docker/quest_relay/run_quest_relay.py
# (--detach to background, --rebuild to force rebuild)
```

On the Quest headset, point the Unity app's ROS-TCP connector at **the laptop's
IP, port 10000** (the laptop is where the relay runs — use the laptop's normal
Wi-Fi/LAN IP that the headset can reach, *not* the 192.168.123 address).

### 2.5 — [LAPTOP] Quest manager (binds 5556, all sources local)

Relay and deploy feedback are both on the laptop → both hosts `localhost`.

```bash
# [LAPTOP]
source .venv_teleop/bin/activate
python gear_sonic/scripts/quest_manager_thread_server.py \
    --relay-host localhost --relay-port 5559 \
    --port 5556 \
    --feedback-host localhost --feedback-port 5557
```

Then: operator in rest pose → press **`s`** to start. `f` toggles fingers, `r`
recalibrates, `p` pause/resume, `q` stop.

### 2.6 — [LAPTOP] Data recorder (optional — for collecting datasets)

Cameras are on the robot → `--camera-host $ROBOT_IP`. The manager (5556) and
deploy (5557) are on the laptop → those stay `localhost`.

```bash
# [LAPTOP]
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the box" \
    --camera-host $ROBOT_IP --camera-port 5555 \
    --sonic-zmq-host localhost --sonic-zmq-port 5556 \
    --state-zmq-host localhost --state-zmq-port 5557 \
    --dataset-name my_session
```

Arm/save episodes from the Quest manager: `c` = start/stop recording, `x` =
abort.

### 2.7 — [LAPTOP] Camera viewer (optional — "see what the robot sees")

```bash
# [LAPTOP]
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host $ROBOT_IP --camera-port 5555
# Window focused: R = start/stop MP4 recording, Q = quit
```

---

## Part 2B (Option B) — Deploy on the ROBOT instead

If you'd rather close the 500 Hz control loop **on the robot's Jetson** (lowest
latency, the recommended way for real runs), the deploy binary moves on-board and
the host flags flip around. The teleop + recording pieces stay on the laptop
(the Quest connects to the laptop, and you review datasets there).

**What moves where:**

| Process | Option A (Part 2) | Option B (here) |
|---|---|---|
| BrainCo hand service | ROBOT | ROBOT |
| Camera server | ROBOT | ROBOT |
| **deploy binary** | LAPTOP | **ROBOT** |
| Quest relay | LAPTOP | LAPTOP |
| Quest manager | LAPTOP | LAPTOP |
| Data recorder | LAPTOP | LAPTOP |
| Camera viewer | LAPTOP | LAPTOP |

**Two things change conceptually:**

1. **DDS is now entirely inside the robot.** The deploy binary, the robot's
   onboard control service, and the BrainCo service are all on `192.168.123.x`
   *inside the Jetson* — `deploy.sh real` auto-detects the internal interface with
   nothing to configure. **The laptop is no longer part of the DDS/control path**,
   so the cable link is no longer carrying the realtime loop.
2. **The ZMQ links now cross the cable** (they were all `localhost` in Option A).
   The manager/recorder on the laptop talk ZMQ to the deploy/camera on the robot.
   So you **still need the laptop on the robot's network** — **Part 1's NIC setup
   is still required**, but now only for these (non-realtime) ZMQ links, not for
   DDS.

Define both IPs. On the **laptop**:

```bash
export ROBOT_IP=192.168.123.164      # the robot — what you ssh to
export LAPTOP_IP=192.168.123.222     # your laptop's static address from Part 1
```

### B.1 — [ROBOT] BrainCo hand service & camera server

Identical to §2.1 and §2.2 — unchanged.

### B.2 — [ROBOT] Deploy binary (now on-board)

First time on the Jetson, make sure build deps + models are present:

```bash
# [ROBOT]
cd <repo> && git lfs pull                 # models/meshes
export TensorRT_ROOT=$HOME/TensorRT       # add to ~/.bashrc; run scripts/install_deps.sh if CUDA/onnx missing
```

Then build + launch. The manager runs on the laptop, so point `--zmq-host` at the
laptop; keep `--output-type all` (or `zmq`) so it publishes the
`g1_debug`/`robot_config` feedback the manager and recorder need:

```bash
# [ROBOT]
cd gear_sonic_deploy
./deploy.sh real --zmq-host $LAPTOP_IP --output-type all
```

`real` auto-detects the Jetson's internal `192.168.123.x` interface — no NIC work
on the robot. Confirm the resolved interface in the banner, wait for
`WAIT_FOR_CONTROL`.

> Tip: `--output-type zmq` (instead of `all`) skips the ROS2 dependency — the
> Quest pipeline only needs the ZMQ `g1_debug` feed. Use it if the Jetson doesn't
> have ROS2 set up.

### B.3 — [LAPTOP] Quest relay

Identical to §2.4 — runs on the laptop; the Quest app still targets the
**laptop's** LAN/Wi-Fi IP : `10000`.

### B.4 — [LAPTOP] Quest manager (feedback now comes from the robot)

The manager still **binds** 5556 locally, but the deploy's `g1_debug` feedback
(5557) is now **on the robot**:

```bash
# [LAPTOP]
source .venv_teleop/bin/activate
python gear_sonic/scripts/quest_manager_thread_server.py \
    --relay-host localhost --relay-port 5559 \
    --port 5556 \
    --feedback-host $ROBOT_IP --feedback-port 5557
```

### B.5 — [LAPTOP] Data recorder (camera & state now on the robot)

Camera server and deploy are both on the robot; only the manager is local:

```bash
# [LAPTOP]
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the box" \
    --camera-host $ROBOT_IP --camera-port 5555 \
    --sonic-zmq-host localhost --sonic-zmq-port 5556 \
    --state-zmq-host $ROBOT_IP --state-zmq-port 5557 \
    --dataset-name my_session
```

### B.6 — [LAPTOP] Camera viewer

Identical to §2.7 (`--camera-host $ROBOT_IP`).

### Option B host-flags cheat sheet

| Process | Runs on | Flag | Value |
|---|---|---|---|
| Deploy (`deploy.sh real`) | ROBOT | `--zmq-host` | `$LAPTOP_IP` |
| Deploy | ROBOT | `--output-type` | `all` (or `zmq`) |
| Quest manager | LAPTOP | `--relay-host` | `localhost` |
| Quest manager | LAPTOP | `--feedback-host` | `$ROBOT_IP` |
| Data recorder | LAPTOP | `--camera-host` | `$ROBOT_IP` |
| Data recorder | LAPTOP | `--sonic-zmq-host` | `localhost` |
| Data recorder | LAPTOP | `--state-zmq-host` | `$ROBOT_IP` |
| Camera viewer | LAPTOP | `--camera-host` | `$ROBOT_IP` |

**Rule of thumb (Option B):** flags that point at the **manager** (`--zmq-host`,
`--sonic-zmq-host`) are `localhost`-relative to the laptop; everything that reads
from the **deploy or camera** (`--feedback-host`, `--state-zmq-host`,
`--camera-host`) points at `$ROBOT_IP`. It's the mirror image of Option A.

> **All-on-robot variant:** you can also run the relay + manager + recorder on the
> Jetson too — then every flag is `localhost`, but the Quest app must target the
> **robot's** IP:`10000` and your datasets live on the Jetson. Simpler flags, less
> convenient for teleop and dataset review.

---

## 3. Host-flags cheat sheet (Option A: deploy on laptop)

| Process | Runs on | Flag(s) you set | Value |
|---|---|---|---|
| BrainCo hand service | ROBOT | `-n` | robot's `192.168.123.x` iface (e.g. `eth0`) |
| Camera server | ROBOT | `--port` | `5555` |
| Deploy (`deploy.sh real`) | LAPTOP | `--zmq-host` | `localhost` |
| Deploy | LAPTOP | `--output-type` | `all` (or `zmq`) |
| Quest relay | LAPTOP | (Quest app target) | laptop's LAN/Wi-Fi IP : `10000` |
| Quest manager | LAPTOP | `--relay-host` | `localhost` |
| Quest manager | LAPTOP | `--feedback-host` | `localhost` |
| Data recorder | LAPTOP | `--camera-host` | `$ROBOT_IP` |
| Data recorder | LAPTOP | `--sonic-zmq-host` | `localhost` |
| Data recorder | LAPTOP | `--state-zmq-host` | `localhost` |
| Camera viewer | LAPTOP | `--camera-host` | `$ROBOT_IP` |

**Rule of thumb:** the *only* `--*-host` that points at the robot is
`--camera-host` (and the deploy's DDS, which you handle via the NIC in Part 1,
not a host flag). Everything else is `localhost` because it runs on the laptop.

---

## 4. Quick "is it working?" checks

- `ping -c3 $ROBOT_IP` from the laptop replies → NIC/cable good (Part 1).
- Deploy banner shows **Resolved interface = your ethernet port** → DDS will
  reach the robot.
- Arms move but **fingers dead** → BrainCo service not up, or it's bound to a
  different interface than the deploy binary (both must be on the same
  `192.168.123.x` / DDS domain 0). Re-check §2.1.
- Robot **never leaves `WAIT_FOR_CONTROL`** → manager not connected / you haven't
  pressed `s`. Since manager+deploy are both local, the ZMQ host is `localhost`;
  confirm both are running.
- Recorder **hangs at startup** → it waits for `robot_config` from the deploy;
  make sure deploy uses `--output-type all`/`zmq` and `--state-zmq-host
  localhost`.

> **Latency note:** running the deploy binary on the laptop means your laptop
> closes the 500 Hz body-control loop *across the cable*. Use a solid wired link
> (Part 1) — do **not** try to do this over Wi-Fi.
