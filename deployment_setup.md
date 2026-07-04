# Deployment Setup — Everything-on-Robot Topology

This is the practical setup guide for the **standard topology**: the entire
real-time teleop path runs **on the robot**, the **Quest connects to the robot's
own WiFi AP**, and the **laptop only records datasets and drives the manager
keyboard over `ssh`**.

**Why this layout:** it kills the latency of the old Quest → laptop → robot path.
Before, every pose crossed WiFi to the laptop, went through the relay + manager
there, then crossed the ethernet cable to the robot — two network hops plus two
app stages on the laptop. Now the relay, the manager (retargeting), the deploy
binary, the camera server, and the hand service all run on the Jetson, so every
ZMQ/DDS link is `localhost` on-board and the Quest is **one WiFi hop** from the
robot.

> **Honest caveat:** the Quest link is still WiFi (to the robot's AP), so radio
> jitter isn't eliminated the way a wire would. Use a **5 GHz** AP with good
> antenna placement. The eventual jitter killer is a wired Quest — see
> **"Future: three-way wired Quest"** at the end.

**Fastest path: the launch scripts.** `scripts/launch_robot_side.sh` brings up the
AP + hand + camera + deploy + relay + manager on the robot;
`scripts/launch_laptop_side.sh` brings up the recorder + viewer on the laptop. Both
bake in every host flag (see **Part 2 — Quick start**). The manual, flag-by-flag
walkthrough follows for reference and troubleshooting.

For the full conceptual reference (architecture, ports, troubleshooting), see
`deployment_report.md`. The one piece of networking you still set up by hand is
**giving your laptop an address on the robot's ethernet subnet** (Part 1) — needed
so the laptop recorder can pull camera/state/manager streams over the cable.

---

## 0. Your topology at a glance

```
   Quest ──WiFi (robot AP, 5GHz)──▶ ROBOT / Jetson  (everything real-time)
                                    ┌──────────────────────────────────────┐
                                    │ setup_ap.sh  → WiFi AP (192.168.55.1) │
                                    │ brainco_hand_service      (DDS)       │
                                    │ camera server             (ZMQ 5555)  │
                                    │ g1_deploy_onnx_ref  (DDS; SUB 5556,   │
                                    │                          PUB 5557)    │
                                    │ quest_relay (Docker)  (TCP 10000,     │
                                    │   +image_relay → Quest;   PUB 5559)   │
                                    │ quest_manager  (SUB 5559 → PUB 5556)  │
                                    └───────────────▲──────────────────────┘
                                                    │ ethernet cable (192.168.123.x)
   LAPTOP (recording + control only)                │  — recorder pulls + ssh
   ┌────────────────────────────────────────────────┴───┐
   │ ssh <robot> ; tmux attach -t g1_robot  → keys s/r/… │
   │ run_data_exporter  (recorder)                       │
   │ run_camera_viewer  (optional)                       │
   └─────────────────────────────────────────────────────┘
```

Every real-time link is `localhost` on the Jetson. Two kinds of traffic still ride
the **ethernet cable** (SSH is a third, unrelated thing on the cable — it does not
carry robot data):

1. **The recorder's ZMQ pulls** — camera (5555), manager pose (5556), and deploy
   state (5557), all bound on the robot, read by the laptop recorder/viewer. Plain
   TCP; just needs the ports reachable over the cable.
2. **SSH** — you `ssh` to the robot to run `launch_robot_side.sh` and to `tmux
   attach` and drive the manager keys.

**The DDS/control loop is now entirely inside the robot** (deploy ⇄ body motors/IMU
⇄ BrainCo hand service, all on `192.168.123.x` inside the Jetson). The laptop is no
longer in the realtime path at all.

---

## Part 1 — The "NIC" thing (laptop address on the robot's ethernet subnet)

You still need this so the laptop recorder can reach the robot's ZMQ ports over the
cable. (It is no longer needed for DDS — that's all on-board now.)

### What a NIC is

**NIC = Network Interface Controller** — the technical name for a network port.
Your laptop's **ethernet jack** is one NIC; its **Wi-Fi card** is another. Each has
a name (`eth0`, `enp0s31f6`, `enx00e04c…`) and can be given an **IP address**.

### Why you set one up by hand

The robot's internals live on the private range **`192.168.123.x`**. Plug a cable
straight into the robot and nobody hands your laptop an address (the robot is not a
router), so it can't talk to the robot's ports. You **manually give your laptop's
ethernet NIC the fixed address `192.168.123.222`.** That's the whole "NIC thing."

> We pick **`.222`** because it's almost certainly unused. Any free number on
> `192.168.123.x` works except the robot's own (commonly `.161`/`.164`).

### Step 1.1 — Find your ethernet NIC's name

Plug the cable into the robot, then on the **laptop**:

```bash
ip -br link
```

The wired one usually starts with `en`, `eth`, or `enx` (Wi-Fi like `wlp3s0` — ignore).
If unsure, unplug the cable and re-run: the line that flips to `DOWN`/disappears is
your ethernet NIC. **Note the name** (examples below use `enp0s31f6`).

### Step 1.2 — Give that NIC a static `192.168.123.x` address

**Method A — temporary (gone after reboot/unplug, simplest):**

```bash
sudo ip addr add 192.168.123.222/24 dev enp0s31f6
sudo ip link set enp0s31f6 up
```

**Method B — persistent (NetworkManager):**

```bash
sudo nmcli connection add type ethernet ifname enp0s31f6 con-name robot \
    ipv4.method manual ipv4.addresses 192.168.123.222/24
sudo nmcli connection up robot
```

Remove later with `sudo nmcli connection down robot`. Don't set a gateway — you
don't want internet routed through the robot.

### Step 1.3 — Verify

```bash
ip -br addr show enp0s31f6        # should list 192.168.123.222/24
ping -c3 <ROBOT_IP>               # e.g. ping -c3 192.168.123.164
```

`<ROBOT_IP>` is the address you `ssh` to. If ping replies, the laptop is on the
robot subnet and the recorder will work.

---

## Part 2 — Quick start with the launch scripts

Set the robot IP once (both scripts default to `ROBOT_IP=192.168.123.164`):

```bash
export ROBOT_IP=192.168.123.164     # what you ssh to (verify: ip -br addr on the robot)
```

### Step 2.0 — Bring up the robot WiFi AP (once)

`ssh` into the robot and start its AP so the Quest can join. This is a **one-shot**
— run it once per boot, **not** on every relaunch (recreating the AP profile drops
the Quest link):

```bash
# [ROBOT] — override SSID/pass/IP via AP_SSID / AP_PASS / AP_IP
./scripts/launch_robot_side.sh ap
#   … or call the script directly:
# gear_sonic_deploy/scripts/setup_ap.sh --ssid g1-teleop --password groot1234 --ip 192.168.55.1
```

Defaults: SSID `g1-teleop`, password `groot1234`, AP IP `192.168.55.1` (5 GHz). The
script preflights AP-mode support (`iw list`) and, if the onboard radio is
client-only, tells you to use a USB WiFi dongle (`--interface <iface>`). Tear the AP
down with `gear_sonic_deploy/scripts/setup_ap.sh down`.

> **Single radio:** while it's an AP, the Jetson can't also be a WiFi *client* for
> internet. During teleop that's fine — the laptop link is the ethernet cable.

On the **Quest**: join WiFi **`g1-teleop`**, then point the Unity app's ROS-TCP
connector at **`192.168.55.1:10000`**. Set the `ImageView` `DebayerMode` to `None`
for the ego-view stream.

### Step 2.1 — Start the robot stack

```bash
# [ROBOT] safety: robot suspended, e-stop in hand — deploy commands the robot immediately
./scripts/launch_robot_side.sh            # tmux 'g1_robot': hand → camera → deploy → relay → manager
```

One tmux window, tiled panes, started in order. The camera defaults to the
RealSense (`EGO_VIEW_CAMERA=oak` / `OAK_SERIAL` to switch). The hand service is
launched manually by default; `HAND_USE_SYSTEMD=1` uses systemd. Tune the manager
with `MANAGER_EXTRA` (e.g. `MANAGER_EXTRA="--static-base --log-latency"`).

### Step 2.2 — Start the laptop stack (recording)

```bash
# [LAPTOP]
./scripts/launch_laptop_side.sh   # tmux 'g1_laptop': recorder + viewer
```

All recorder sources (camera 5555, manager 5556, deploy state 5557) are at
`$ROBOT_IP` over the cable. Tune with `TASK_PROMPT` / `DATASET_NAME`.

### Step 2.3 — Drive the manager (control commands, operator-side)

The manager runs headless on the robot, but its keyboard state machine needs a real
terminal — **SSH provides one**. From the laptop:

```bash
# [LAPTOP]
ssh <robot>
tmux attach -t g1_robot        # focus the 'manager' pane (Ctrl-b <arrow>)
```

In the manager pane: operator in rest pose → **`s`** to start (press `s` twice: ramp
→ calibrate). Then `f` toggles fingers, `r` recalibrates, `p` pause/resume,
`c` start/stop recording, `x` abort recording, `q` stop. This SSH session is the
"something on the laptop that sends setup and kill commands" — it's active only for
the keys, never in the data path.

**tmux basics**: `Ctrl-b <arrow>` moves between panes; `Ctrl-b z` zooms; `Ctrl-b d`
detaches; `tmux attach -t g1_robot` re-attaches. Tear down with
`./scripts/launch_robot_side.sh kill` (or `... laptop ... kill`). A pane whose
component exits stays open showing `[… exited]`; press Enter to close it.

### Single-component / debug mode

Each script accepts one component name to run it alone in the foreground (handy for
restarting one piece). Run with `-h` for usage + resolved config:

```bash
# robot: ap | hand | camera | deploy | relay | manager      laptop: recorder | viewer
./scripts/launch_robot_side.sh deploy     # just (re)start deploy here
./scripts/launch_laptop_side.sh recorder  # just (re)start the recorder here
```

The rest of this document is the manual, flag-by-flag reference the scripts
automate — read it to understand or debug what a component runs.

---

## Part 3 — Manual reference: every command, hosts filled in

Open **two kinds of terminals**:

- **[ROBOT]** = a terminal `ssh`'d into the robot.
- **[LAPTOP]** = a local terminal on your laptop.

Set the robot IP once on the **laptop**:

```bash
export ROBOT_IP=192.168.123.164        # whatever you ssh to
```

> Reminder on hosts: a `--*-host` flag is always **the IP of the machine that
> *binds* (opens) that port**. On the robot everything binds on the robot, so all
> the robot-side `--*-host` flags are `localhost`. The laptop recorder reads from
> the robot, so its source hosts are `$ROBOT_IP`.

### 3.0 — Safety first

Robot powered but **suspended/safe**, **e-stop in your hand**. `deploy.sh` runs
unattended and starts commanding the robot immediately.

### 3.1 — [ROBOT] WiFi AP

```bash
# [ROBOT]
gear_sonic_deploy/scripts/setup_ap.sh --ssid g1-teleop --password groot1234 --ip 192.168.55.1
# down:  gear_sonic_deploy/scripts/setup_ap.sh down
```

Quest joins `g1-teleop`, targets `192.168.55.1:10000`.

### 3.2 — [ROBOT] BrainCo hand service (must be up first)

DDS bridge for the hands, on the robot's internal `192.168.123.x` subnet.

```bash
# [ROBOT] — systemd (recommended):
sudo systemctl status brainco_hand.service          # green/active = good
sudo systemctl restart brainco_hand.service         # if needed
# --- OR by hand: ---
cd <repo>/gear_sonic_deploy/thirdparty/brainco_hand_service/bin
sudo ./brainco_hand_server -n <robot_iface>         # e.g. -n eth0

# sanity: cycle the hands open/closed
sudo ./test_brainco_hand_server left
sudo ./test_brainco_hand_server right
```

### 3.3 — [ROBOT] Camera server (binds ZMQ 5555)

```bash
# [ROBOT]
source .venv_data_collection/bin/activate
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera realsense \
    --port 5555
# (--ego-view-camera oak --ego-view-device-id <OAK_SERIAL>, or usb / an .mp4 path)
```

### 3.4 — [ROBOT] Deploy binary (DDS to robot + binds 5557, subscribes 5556)

The manager runs on the robot too, so `--zmq-host localhost`. `--output-type all`
(or `zmq`) makes it publish the `g1_debug`/`robot_config` feedback the manager and
recorder need.

```bash
# [ROBOT]
cd gear_sonic_deploy
./deploy.sh real --zmq-host localhost --output-type all
```

`real` auto-detects the Jetson's internal `192.168.123.x` interface — no NIC work on
the robot. Confirm the resolved interface in the banner; wait for
`WAIT_FOR_CONTROL`.

> Tip: `--output-type zmq` skips the ROS2 dependency — the Quest pipeline only needs
> the ZMQ `g1_debug` feed. Use it if the Jetson doesn't have ROS2 set up.

### 3.5 — [ROBOT] Quest relay + ego-view image relay (binds TCP 10000 + ZMQ 5559)

Runs on the robot with **host networking** (`--network-host`): no Docker-NAT hop,
and `localhost` reaches the on-board camera server. `--camera-host localhost`
starts the image relay, which republishes the ego-view as a ROS1 `CompressedImage`
the Quest's Unity `ImageView` head-locks.

```bash
# [ROBOT] — from repo root
python3 gear_sonic_deploy/docker/quest_relay/run_quest_relay.py \
    --network-host \
    --camera-host localhost --camera-port 5555 --image-fps 30
# (--rebuild to force a rebuild; first build on the Jetson pulls the arm64 base and is slow)
```

Omit `--camera-host` for the plain relay (no camera in the headset). The Quest app
targets the **robot AP IP : 10000** (`192.168.55.1:10000`).

### 3.6 — [ROBOT] Quest manager (binds 5556; all sources local)

```bash
# [ROBOT]
source .venv_teleop/bin/activate
python gear_sonic/scripts/quest_manager_thread_server.py \
    --relay-host localhost --relay-port 5559 \
    --port 5556 \
    --feedback-host localhost --feedback-port 5557
# add --static-base / --log-latency / --smooth-tau / --pos-scale as needed
```

Drive it over SSH (§2.3): `s` start, `f` fingers, `r` recalibrate, `p` pause,
`c` record, `x` abort, `q` stop.

> **Jetson dependencies:** the manager imports the G1 pinocchio model at startup, so
> the robot's `.venv_teleop` needs `pinocchio` (aarch64) plus `numpy/scipy/pyzmq/
> msgpack/pyyaml` and, for the optimization finger path, `dex_retargeting`+`nlopt`+
> editable `brainco_retargeting`. If those won't install, `--np-retarget` uses the
> pure-numpy finger path — but pinocchio is still required.

### 3.7 — [LAPTOP] Data recorder (all sources on the robot)

Camera (5555), manager pose (5556), and deploy state (5557) are all on the robot:

```bash
# [LAPTOP]
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the box" \
    --camera-host $ROBOT_IP --camera-port 5555 \
    --sonic-zmq-host $ROBOT_IP --sonic-zmq-port 5556 \
    --state-zmq-host $ROBOT_IP --state-zmq-port 5557 \
    --dataset-name my_session
```

Arm/save episodes from the manager (over SSH): `c` = start/stop, `x` = abort.

### 3.8 — [LAPTOP] Camera viewer (optional)

```bash
# [LAPTOP]
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host $ROBOT_IP --camera-port 5555
# Window focused: R = start/stop MP4 recording, Q = quit
```

---

## Host-flags cheat sheet

| Process | Runs on | Flag(s) | Value |
|---|---|---|---|
| WiFi AP (`setup_ap.sh`) | ROBOT | `--ssid/--password/--ip` | `g1-teleop` / … / `192.168.55.1` |
| BrainCo hand service | ROBOT | `-n` | robot's `192.168.123.x` iface |
| Camera server | ROBOT | `--port` | `5555` |
| Deploy (`deploy.sh real`) | ROBOT | `--zmq-host` / `--output-type` | `localhost` / `all` (or `zmq`) |
| Quest relay | ROBOT | `--network-host --camera-host` | (flag) / `localhost` |
| Quest relay | — | Quest app target | robot **AP** IP : `10000` |
| Quest manager | ROBOT | `--relay-host` / `--feedback-host` | `localhost` / `localhost` |
| Data recorder | LAPTOP | `--camera/--sonic/--state-zmq-host` | all `$ROBOT_IP` |
| Camera viewer | LAPTOP | `--camera-host` | `$ROBOT_IP` |
| Manager control | LAPTOP | (ssh) | `ssh <robot> ; tmux attach -t g1_robot` |

**Rule of thumb:** on the robot everything is `localhost`; on the laptop every
source host is `$ROBOT_IP`. The Quest targets the robot's **AP** IP (not the
`192.168.123.x` address, and not the laptop).

---

## Quick "is it working?" checks

- Quest sees SSID `g1-teleop`, joins, and Unity connects to `192.168.55.1:10000` →
  AP + relay good. If not: check `setup_ap.sh` succeeded (`iw list` AP support) and
  the relay pane is up.
- `ping -c3 $ROBOT_IP` from the laptop replies → NIC/cable good (Part 1) → recorder
  can reach the robot.
- Deploy banner shows **Resolved interface = the Jetson's internal `192.168.123.x`
  port** → DDS reaches the body + hands.
- Arms move but **fingers dead** → BrainCo service not up, or on a different
  interface/DDS domain than deploy. Re-check §3.2.
- Robot **never leaves `WAIT_FOR_CONTROL`** → manager not connected or you haven't
  pressed `s`. Everything is on the robot now, so the manager's `--relay-host`/
  `--feedback-host` should be `localhost`; confirm the relay + manager panes are up
  and you're attached over SSH to press `s`.
- Recorder **hangs at startup** → it waits for `robot_config` from the deploy;
  ensure deploy uses `--output-type all`/`zmq` and the recorder's `--state-zmq-host`
  is `$ROBOT_IP`.
- Ego-view **not in headset** → relay started without `--camera-host`, or Unity
  `DebayerMode` ≠ `None`, or the camera server isn't up.

---

## Future: three-way wired Quest (lowest jitter)

The remaining WiFi is the **Quest ↔ robot AP** link. A wired Quest removes it
entirely. Options, easiest first:

- **USB-C ethernet on the Quest + a small gigabit switch** shared with the robot
  (and optionally the laptop), all on one subnet: the Quest targets the robot
  directly over the wire, the AP is no longer needed, and the laptop stays only for
  recording. Lowest latency and jitter.
- **Quest tethered to the Jetson** via USB gadget/RNDIS networking (fiddlier; a
  switch is cleaner).

When a wired setup lands, point the Quest's Unity app at the robot's wired IP:`10000`
instead of the AP IP, and skip `setup_ap.sh`. Everything else (relay + manager +
deploy + camera on the robot, recorder on the laptop) is unchanged.
