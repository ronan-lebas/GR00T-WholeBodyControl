# Running Quest teleop on the real G1 — start here

This is the entry point for running the BrainCo-handed G1 under Meta Quest
teleoperation on **real hardware**.

**Branch: `robot-deploy`.**

```bash
git clone git@github.com:ronan-lebas/GR00T-WholeBodyControl.git && cd GR00T-WholeBodyControl
git checkout robot-deploy
git submodule update --init --recursive
git lfs pull
```

The checkpoints are **not** in the repo — see §2(c). Install reference is
upstream: `docs/source/getting_started/installation_deploy.md` and
`download_models.md`.

> **Warning:** Large parts of this stack have not run on hardware since
> 2026-07-01. Rehearse in sim first (§3), and read "Known risks" (§5) and
> "Network gotchas" (§6) before the first session.

---

## 1. Topology

Everything real-time runs **on the robot**. The Quest is **cabled** to the robot.
The laptop only records datasets and drives the manager keyboard over `ssh`.

```
  Quest ──USB-C eth adapter──cable──▶ ROBOT / Jetson          LAPTOP
                                     ┌───────────────────┐   ┌──────────────────┐
                                     │ quest_relay   [D] │   │ run_data_exporter│
                                     │ quest_manager     │◀──┤ run_camera_viewer│
                                     │ g1_deploy_onnx[D] │   │ ssh + tmux attach│
                                     │ camera server     │   └──────────────────┘
                                     │ brainco_hand_svc  │      ▲
                                     └───────────────────┘      │ ethernet cable
                                              └─────────────────┘ (192.168.123.x)
```

Two cables, no WiFi:

| Link | Subnet | Carries |
|---|---|---|
| Quest ↔ robot | `192.168.77.x` | Unity ROS-TCP (10000), ego-view back to the headset |
| Laptop ↔ robot | `192.168.123.x` | recorder/viewer ZMQ pulls (5555/5556/5557) + `ssh` |

`[D]` = runs in Docker. The SONIC deploy stack (`g1_deploy_onnx_ref` + its
CUDA/TensorRT/ONNX-Runtime toolchain) runs inside
`gear_sonic_deploy/docker/run-ros2-dev.sh`, and the Quest relay runs in its own
`ros:noetic` container. Everything else runs on the host in a venv.

---

## 2. One-time setup

### On the robot (Jetson)

Upstream reference: **`docs/source/getting_started/installation_deploy.md`**
(prerequisites, native vs. Docker setup) and
**`docs/source/getting_started/download_models.md`**. This section only covers
what is specific to this fork's robot setup.

**(a) TensorRT.** Make sure the right version of TensorRT is installed on the robot.

**(b) Download the policy + planner checkpoints from Hugging Face.** They are
*not* in the repo and `git lfs pull` does not fetch them — the deploy binary
exits with "Missing file" without them:

```bash
uv run --no-project --with huggingface_hub python download_from_hf.py
```

That lands `model_encoder.onnx`, `model_decoder.onnx`,
`observation_config.yaml` and `planner_sonic.onnx`. Alternative checkpoints:
`--low-latency` or `--sonic-v1-1` (see `docs/source/getting_started/download_models.md`;
both need the matching `--obs-config`, passed via `DEPLOY_EXTRA`). Do **not** use
`--training` — that pulls a ~30 GB SMPL dataset you don't need on the robot.

**(c) Build the container image once**:

```bash
gear_sonic_deploy/docker/run-ros2-dev.sh --host-net    # exit the shell once it comes up
```

**(d) The two host-side venvs** (camera server + Quest manager run outside the
container):

```bash
bash install_scripts/install_data_collection.sh  # .venv_data_collection (camera server)
bash install_scripts/install_pico.sh             # .venv_teleop (quest manager + retargeting)
```

The BrainCo hand bridge also runs on the host — it owns the USB serial ports and
must be up before the deploy binary:

```bash
# submodule; needs unitree_sdk2 installed system-wide first
cd gear_sonic_deploy/thirdparty/brainco_hand_service
mkdir -p build && cd build && cmake .. && make -j6
```

Test the hands standalone before touching the control stack:

```bash
cd gear_sonic_deploy/thirdparty/brainco_hand_service/bin
sudo ./brainco_hand_server
sudo ./test_brainco_hand_server left          # fingers should fist + open
sudo ./test_brainco_hand_server right
```

### On the laptop

```bash
bash install_scripts/install_data_collection.sh   # .venv_data_collection
```

Give the laptop's ethernet NIC an address on the robot's subnet (the robot is not
a DHCP server):

```bash
ip -br link                                   # find your wired iface, e.g. enp0s31f6
sudo nmcli connection add type ethernet ifname enp0s31f6 con-name robot \
    ipv4.method manual ipv4.addresses 192.168.123.222/24
sudo nmcli connection up robot
ping -c3 192.168.123.164                      # the robot — whatever you ssh to
```

---

## 3. In sim

The MuJoCo stack runs the same deploy
binary, manager, relay and recorder, so it exercises everything except the
hardware itself.

```bash
bash install_scripts/install_mujoco_sim.sh    # .venv_sim
uv run --no-project --with huggingface_hub python download_from_hf.py
./scripts/launch_sim_setup.sh
```

### 3.1 — Robot launcher against the sim

On a laptop with **rootful** Docker you can run the *robot-side* launcher itself,
with MuJoCo standing in for the robot. This covers `run-ros2-dev.sh --host-net`,
the relay under `--network-host`, the on-robot manager, the container auto-type
and its `[Y/n]` gate, and the `--input-type zmq_manager` path — i.e. nearly
everything the first robot session depends on.


**Attaching the Quest over USB (`adb reverse`).** 
```bash
adb devices           # headset must show 'device', not 'unauthorized'
adb reverse tcp:10000 tcp:10000
```

The Quest app then should target **`127.0.0.1:10000`**.



```bash
# sim in one terminal (the fake robot, on loopback)
./scripts/launch_sim_setup.sh sim

# the robot-side launcher in another, pointed at the sim
DEPLOY_TARGET=sim ./scripts/launch_robot_side.sh relay
DEPLOY_TARGET=sim ./scripts/launch_robot_side.sh manager
DEPLOY_TARGET=sim ./scripts/launch_robot_side.sh deploy
source scripts/setup_env.sh && ./deploy.sh sim --input-type zmq_manager --zmq-host localhost --output-type all # in the container
```

**No headset? Two ways to drive it:**

```bash
./scripts/launch_sim_setup.sh --mock-quest       # synthetic pose stream
./scripts/launch_sim_setup.sh --replay-quest     # replay a recorded NPZ
```

What to confirm before touching hardware:

1. `s` twice → the robot tracks your wrists; `p`, `f`, `r`, `q` behave as §4.4 says.
2. `c` starts/stops an episode and `x` aborts it, and a dataset lands on disk.

## 4. Running a session on the robot

### 4.1 — Wire up the Quest (once per boot)

Plug the adapter into the Quest and the cable into the robot's USB-ethernet
adapter, then on the **robot**:

```bash
./scripts/launch_robot_side.sh wire
```

This gives the robot side `192.168.77.1/24` with NetworkManager serving DHCP +
NAT, so the headset gets an address by itself.

Verify the headset got a lease: `ip neigh show dev <iface>`.

In the Quest Unity app, point the ROS-TCP connector at **`192.168.77.1:10000`**.

### 4.2 — Start the robot stack

```bash
# [ROBOT]
./scripts/launch_robot_side.sh
```

One tmux session `g1_robot` with five labelled panes started in order: hand →
camera → deploy → relay → manager.

The **deploy pane** does three things by itself:

1. starts the ROS2 container — `docker/run-ros2-dev.sh --host-net`;
2. waits for its `Relays active` banner;
3. types the build+launch command into the container shell:

```bash
source scripts/setup_env.sh && ./deploy.sh real --zmq-host localhost --output-type all
```

Then it stops at `deploy.sh`'s own prompt:

```
Proceed with deployment? [Y/n]:
```

**Type `y` there when you are ready for the robot to be commanded.** That prompt
is deliberately not bypassed on the robot.

`--host-net` is mandatory here: the deploy binary's DDS has to reach the real
`192.168.123.x` interface. The container's bridge default exists for rootless
Docker on the lab machine (sim) and cannot carry it.

Two slow first-runs to expect, neither of them a hang: building the container
image, then the TensorRT engine build from the ONNX (several minutes, cached
afterwards — the cache lives next to the ONNX in the bind-mounted repo, so it
survives container restarts).

The manager defaults to `MANAGER_EXTRA="--static-base"` — **arms and hands only,
no walking, no turn-in-place, no crouch**. See §5 to relax that.

### 4.3 — Start the laptop side

```bash
# [LAPTOP]
export ROBOT_IP=192.168.123.164
./scripts/launch_laptop_side.sh            # tmux 'g1_laptop': recorder + viewer
```

The recorder passes `--hand-type brainco` (see Known risks #3). Set
`TASK_PROMPT=...` and `DATASET_NAME=...` before launching.

### 4.4 — Drive it


| Key | Action |
|---|---|
| `s` | 1st press: start policy + ramp to the calibration pose. 2nd press: countdown → calibrate → teleop |
| `r` | recalibrate — **ramps the robot back to the reference pose first** (3 s), then counts down |
| `p` | pause / resume teleop |
| `f` | toggle finger tracking |
| `c` / `x` | start-stop recording an episode / abort it |
| `-` / `=` | crouch deeper / stand up (works whether or not `--enable-crouch` is set) |
| `q` | stop (sends policy STOP) |
| `b`, `0` | **simulator only — do not press on hardware** (see Known risks #4) |

Tear down with `./scripts/launch_robot_side.sh kill`.

To stop cleanly: `q` in the manager, then Ctrl-C the deploy pane (it damps down).

---

## 5. Known risks — read before the first session

The last end-to-end validation of **body motion** on the real robot was
**2026-07-01** (tag `full-teleoperation-quest-works-on-robot`, commit `9f624c5`).
Everything below was written after that and has only been exercised in MuJoCo, or
on hardware for the hands/recording path only.

1. **Wrist-target pipeline was re-tuned.** `--smooth-tau` dropped 0.05 → 0.02,
   `--pos-scale` is 0.7, and the ZMQ path was reworked (conflate +
   publish-on-receipt). These change what the policy receives every tick. If
   tracking feels wrong, `--smooth-tau 0.05` is the old behaviour.
2. **Locomotion is OFF by default on the robot launcher** (`--static-base`), and
   **head-driven crouch is off in the manager itself** (`--enable-crouch` turns it
   on).
   With it on, the operator's head dropping below the calibration height commands
   an `IDEL_SQUAT` base height,
   and while squatting it cannot walk. It has only ever run in MuJoCo. The
   `-`/`=` keyboard trim crouches regardless, which is the controlled way to try
   it first.
3. **Recorder hand units.** `run_data_exporter.py` defaults to `--hand-type
   dex3`; on BrainCo the deploy forwards hand `q`/action normalized to `[0,1]`,
   which would land in columns that are radians for every other joint.
   `launch_laptop_side.sh` now passes `--hand-type brainco` — keep it if you
   invoke the recorder by hand.
4. **`b` and `0` are sim-only.** `0` makes the manager send a `reanchor` to the
   deploy, which sets `play = false`, parks on a motion snapshot and **blocks up
   to 5 s** before resuming. It exists for MuJoCo scene resets. On hardware it
   would freeze tracking mid-session.
5. **`r` makes the robot go to reference pose.**
6. **The script to wire Quest up is untested.** `setup_quest_wire.sh` is new.

### Suggested bring-up order

`launch_robot_side.sh` defaults to `--static-base`, so out of the box the robot
moves **arms and hands only**.

1. Arms/hands only — the default, no `MANAGER_EXTRA` needed.
2. Add turn-in-place: `MANAGER_EXTRA="--disable-walk"`.
3. Add walking: `MANAGER_EXTRA=""` (explicitly empty — this overrides the default).
4. Try crouch with the `-`/`=` keys first, then `MANAGER_EXTRA="--enable-crouch"`.

The sim launcher defaults the same way, except it also passes `--enable-crouch`.

---

## 6. Potential issues

**1. The hand service and the deploy binary must be on the same interface.**
`launch_robot_side.sh` starts `brainco_hand_server` with **no `-n`** unless you
set `ROBOT_IFACE`, so it uses its own default. If that differs from the
interface deploy resolved, `rt/brainco/*` never meets and the arms move with dead
fingers. Set it explicitly:
`ROBOT_IFACE=<192.168.123.x iface> ./scripts/launch_robot_side.sh`.

**2. Both containers need host networking, for different reasons.** The deploy
container needs `--host-net` so DDS sees the real robot LAN (its bridge default
exists for rootless Docker on the lab machine). The relay container needs
`--network-host` so `localhost:5555` reaches the on-board camera server and so
port 10000 is bound on the Quest wire rather than behind Docker NAT. The launcher
passes both; if you start either by hand, don't drop them.

**3. Stale containers block a restart.** Both containers run under fixed names
(`g1-deploy-dev`, `quest-relay`). If a pane was killed uncleanly you'll get
"name already in use":

```bash
docker rm -f g1-deploy-dev quest-relay
./scripts/launch_robot_side.sh kill      # then relaunch
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Arms move, fingers dead | `brainco_hand_service` down, or on a different iface/DDS domain than deploy | `systemctl status brainco_hand`; both must be on the robot's `192.168.123.x` iface, domain 0 |
| Never leaves `WAIT_FOR_CONTROL` | manager hasn't sent START | press `s` in the manager pane; check it is connected to the relay |
| `Missing file` at deploy startup | checkpoints never downloaded | `python download_from_hf.py` (§2c) — `git lfs pull` does not fetch them |
| Recorder hangs at startup | waiting for `robot_config` from deploy | deploy needs `--output-type all`/`zmq`; check `--state-zmq-host` |
| Deploy talks DDS, robot ignores it | wrong interface resolved | check the "Resolved interface" banner is the internal `192.168.123.x` port |
| Slow first start | rebuilding TRT engines from ONNX | expected on new hardware; cached afterwards |
| Deploy container up, but no DDS to the robot | started without `--host-net` | the launcher passes it; if running by hand, `run-ros2-dev.sh --host-net` |
| Deploy pane sits at a container shell | the auto-type watcher missed the ready marker | paste the printed `source scripts/setup_env.sh && ./deploy.sh …` yourself |
| Deploy pane waits at `[Y/n]` | expected — this is the safety gate | type `y` once the robot is ready to go |

