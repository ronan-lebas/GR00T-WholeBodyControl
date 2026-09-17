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

**(a) Docker with the NVIDIA runtime** — JetPack provides it:

```bash
docker info | grep -i -A3 runtime        # expect 'nvidia' among the runtimes
sudo usermod -aG docker $USER            # then log out/in, if not already done
```

**(b) TensorRT.** Nothing to install on a Jetson: if
`/usr/lib/aarch64-linux-gnu/libnvinfer.so` exists (it does under JetPack), the
container stages JetPack's own TensorRT into `/opt/TensorRT` automatically. Only
if that file is missing do you need a standalone install plus
`export TensorRT_ROOT=$HOME/TensorRT` in `~/.bashrc` — the container mounts it.
CUDA, ONNX Runtime, `just` and CMake all live **inside** the container; you do
not install them on the Jetson.

**(c) Download the policy + planner checkpoints from Hugging Face.** They are
*not* in the repo and `git lfs pull` does not fetch them — the deploy binary
exits with "Missing file" without them:

```bash
pip install huggingface_hub                  # if not present
python download_from_hf.py                   # → gear_sonic_deploy/policy/release/
                                             #   + planner/target_vel/V2/
```

That lands `model_encoder.onnx`, `model_decoder.onnx`,
`observation_config.yaml` and `planner_sonic.onnx`. Alternative checkpoints:
`--low-latency` or `--sonic-v1-1` (see `docs/source/getting_started/download_models.md`;
both need the matching `--obs-config`, passed via `DEPLOY_EXTRA`). Do **not** use
`--training` — that pulls a ~30 GB SMPL dataset you don't need on the robot.

**(d) Build the container image once** (slow — arm64 base pull):

```bash
gear_sonic_deploy/docker/run-ros2-dev.sh --host-net    # exit the shell once it comes up
```

**(e) The two host-side venvs** (camera server + Quest manager run outside the
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
cd .. && bash setup_autostart.sh             # installs brainco_hand.service
```

Confirm the deploy binary is built for BrainCo (it ships this way):

```bash
grep USE_BRAINCO_HANDS gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/hand_config.hpp
# → #define USE_BRAINCO_HANDS 1     (compile-time; changing it requires a rebuild)
```

Test the hands standalone **before** touching the control stack:

```bash
cd gear_sonic_deploy/thirdparty/brainco_hand_service/bin
sudo ./brainco_hand_server -n <robot_iface>   # iface holding the 192.168.123.x address
sudo ./test_brainco_hand_server left          # fingers should fist + open
sudo ./test_brainco_hand_server right
```

Do not continue until both hands cycle.

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

## 3. Rehearse in simulation first

Do this before you go near the robot. The MuJoCo stack runs the **same** deploy
binary, manager, relay and recorder, so it exercises everything except the
hardware itself — and it is where you learn the keyboard state machine without a
robot in front of you.

Run it on a machine with a GPU (your workstation, not the Jetson):

```bash
bash install_scripts/install_mujoco_sim.sh    # .venv_sim
python download_from_hf.py                    # same checkpoints as the robot
./scripts/launch_sim_setup.sh -n              # -n prints the plan without running
./scripts/launch_sim_setup.sh                 # tmux 'g1_sim', all panes
```

### 3.1 — Dress rehearsal: the robot launcher against the sim

If your machine has **rootful** Docker (not rootless) and the Quest cable, you can
exercise the *robot-side* launcher itself with MuJoCo standing in for the robot —
this covers `setup_quest_wire.sh`, `run-ros2-dev.sh --host-net`, the relay with
`--network-host`, and the on-robot manager, i.e. nearly everything the first robot
session depends on:

```bash
# sim in one terminal (provides the fake robot on loopback)
./scripts/launch_sim_setup.sh sim

# the robot-side launcher in another, pointed at the sim
DEPLOY_TARGET=sim ./scripts/launch_robot_side.sh wire     # real Quest cable
DEPLOY_TARGET=sim ./scripts/launch_robot_side.sh relay
DEPLOY_TARGET=sim ./scripts/launch_robot_side.sh manager
DEPLOY_TARGET=sim ./scripts/launch_robot_side.sh deploy
```

Skip the `hand` component (no BrainCo hardware) and start `camera` only if one is
plugged in. What this does **not** cover: DDS to real motors and hands, and the
`192.168.123.x` laptop cable.

> **Rootless Docker cannot do this.** Under rootless, `--network host` puts the
> container in the rootless *network namespace*, not the real host — verified:
> the host's `enp3s0` is invisible and you get a `tap0 10.0.2.100` instead. That
> is why the bridge + port-map setup exists (commit `9fede6d`) and why the sim
> launcher does not pass `--host-net`. On a rootless machine, test the sim stack
> only (§3) and leave `--host-net` for the robot or a rootful laptop.

The deploy pane behaves exactly like the robot's — container, then `deploy.sh`,
then the `[Y/n]` prompt — except it runs `deploy.sh sim` with `--yes` (no robot
to endanger) and on bridge networking.

**No headset? Two ways to drive it:**

```bash
./scripts/launch_sim_setup.sh --mock-quest       # synthetic pose stream
./scripts/launch_sim_setup.sh --replay-quest     # replay a recorded NPZ
```

**With a headset**, point the Quest app at that machine's LAN IP `:10000` (in sim
the relay runs on the workstation, not behind the robot's wire).

What to confirm before touching hardware:

1. `s` twice → the robot tracks your wrists; `p`, `f`, `r`, `q` behave as §4.4 says.
2. `c` starts/stops an episode and `x` aborts it, and a dataset lands on disk.
3. You can read the deploy banner — the resolved interface, the checkpoint paths,
   and `Encoder ... Input dimension` matching your obs config.

Two differences from the robot that will *not* show up in sim: DDS to real
motors/hands, and every network hop in §6. Sim proves the software, not the
wiring.

---

## 4. Running a session on the robot

### 4.1 — Wire up the Quest (once per boot)

Plug the adapter into the Quest and the cable into the robot's USB-ethernet
adapter, then on the **robot**:

```bash
./scripts/launch_robot_side.sh wire
```

This gives the robot side `192.168.77.1/24` with NetworkManager serving DHCP +
NAT, so the headset gets an address by itself — nothing to configure on the
Quest. The script refuses any interface that already has an IPv4 address, so it
cannot steal the DDS LAN or the laptop cable; pass `--interface enxXXXX` if
auto-detection is ambiguous.

Verify the headset got a lease: `ip neigh show dev <iface>`.

In the Quest Unity app, point the ROS-TCP connector at **`192.168.77.1:10000`**
and set `ImageView` `DebayerMode` to `None`.

### 4.2 — Start the robot stack

**Safety first: robot suspended, e-stop in hand.** `deploy.sh` builds and then
starts commanding the robot; its `[Y/n]` prompt is the only gate, and you answer
it by hand (§4.2 below) — nothing passes `--yes` on the robot.

```bash
# [ROBOT]
./scripts/launch_robot_side.sh
```

One tmux session `g1_robot` with five labelled panes started in order: hand →
camera → deploy → relay → manager. Wait for deploy to reach `WAIT_FOR_CONTROL`.

The **deploy pane** does three things by itself, then hands you the wheel:

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
is deliberately not bypassed on the robot. Outside tmux the command is printed
rather than typed — paste it into the container shell yourself.

`--host-net` is mandatory here: the deploy binary's DDS has to reach the real
`192.168.123.x` interface. The container's bridge default exists for rootless
Docker on the lab machine (sim) and cannot carry it.

Two slow first-runs to expect, neither of them a hang: building the container
image, then the TensorRT engine build from the ONNX (several minutes, cached
afterwards — the cache lives next to the ONNX in the bind-mounted repo, so it
survives container restarts).

Useful overrides: `EGO_VIEW_CAMERA=oak OAK_SERIAL=...`, `OUTPUT_TYPE=zmq` (skips
ROS2), `MANAGER_EXTRA="--static-base"`, and `DEPLOY_EXTRA="--cp
policy/sonic_v1_1/model --obs-config policy/sonic_v1_1/observation_config.yaml"`
to switch checkpoint (same knob as the sim launcher).

### 4.3 — Start the laptop side

```bash
# [LAPTOP]
export ROBOT_IP=192.168.123.164
./scripts/launch_laptop_side.sh            # tmux 'g1_laptop': recorder + viewer
```

The recorder passes `--hand-type brainco` (see Known risks #3). Set
`TASK_PROMPT=...` and `DATASET_NAME=...` before launching.

### 4.4 — Drive it

The manager's keyboard lives **on the robot**, so drive it over ssh:

```bash
# [LAPTOP]
ssh <robot>
tmux attach -t g1_robot          # then focus the 'manager' pane (Ctrl-b <arrow>)
```

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

tmux: `Ctrl-b <arrow>` to move between panes, `Ctrl-b z` to zoom one, `Ctrl-b d`
to detach. Tear down with `./scripts/launch_robot_side.sh kill`.

To stop cleanly: `q` in the manager, then Ctrl-C the deploy pane (it damps down).

---

## 5. Known risks — read before the first session

The last end-to-end validation of **body motion** on the real robot was
**2026-07-01** (tag `full-teleoperation-quest-works-on-robot`, commit `9f624c5`).
Everything below was written after that and has only been exercised in MuJoCo, or
on hardware for the hands/recording path only. None of it is known-broken — it is
untested.

1. **Wrist-target pipeline was re-tuned.** `--smooth-tau` dropped 0.05 → 0.02,
   `--pos-scale` is 0.7, and the ZMQ path was reworked (conflate +
   publish-on-receipt). These change what the policy receives every tick. If
   tracking feels wrong, `--smooth-tau 0.05` is the old behaviour.
2. **Head-driven crouch is OFF by default** (`--enable-crouch` turns it on).
   With it on, the operator's head dropping below the calibration height commands
   an `IDEL_SQUAT` base height — anyone who ducks or leans makes the robot squat,
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
5. **`r` behaves differently than it used to.** It now eases the robot back to
   the reference pose over `--calib-ramp-sec` (3 s) before the countdown — a real
   arm motion the old build did not perform.
6. **The everything-on-robot topology has never been run end to end.** The relay
   and manager moving onto the Jetson was written in July and only the Docker
   image build was ever exercised there (hence the `pyzmq==25.1.2` pin). Expect
   to debug the relay container and venv paths on the Jetson.
   `run-ros2-dev.sh --host-net` is also new — the container has only ever been
   run in bridge mode, against the MuJoCo sim.
7. **`--input-type` was wrong on the robot launcher until now.** `deploy.sh`
   defaults to `manager`, which builds an `InterfaceManager` around a
   `ZMQEndpointInterface` — that subscribes to the `pose` topic only, so the
   quest manager's `command` (START) and `planner` (VR targets) were never read
   and the robot would have sat in `WAIT_FOR_CONTROL` forever. The launcher now
   passes `--input-type zmq_manager`, matching the sim launcher. Mentioned
   because `deployment_report.md` §4.5 still claims `manager` "subscribes to the
   manager's command/planner topics" — it does not.
8. **The wired Quest link is untested.** `setup_quest_wire.sh` is new. Its
   refusal paths are verified, but the `up` path has never run against a real
   adapter. If `nmcli` fights you, the manual equivalent is in the script header.

### Suggested bring-up order

Robot suspended and e-stopped throughout, relaxing one constraint at a time:

1. Hands only, no body: `MANAGER_EXTRA="--static-base"`.
2. Add turn-in-place: `MANAGER_EXTRA="--disable-walk"`.
3. Add walking: no extra flags.
4. Try crouch with the `-`/`=` keys only, then `MANAGER_EXTRA="--enable-crouch"`.
5. Only then record datasets.

---

## 6. Network gotchas

Most of what can go wrong on the robot is networking. These are the specific
traps in this setup, roughly in order of how likely they are to bite.

**1. The Quest gets an IP but Unity still won't connect.** This is the most
likely failure. `ipv4.method shared` makes NetworkManager run DHCP and enable
`net.ipv4.ip_forward` + masquerade, but it does **not** guarantee that inbound
TCP 10000 to the robot is accepted — if `firewalld`/`ufw`/`nftables` is active on
the Jetson, the DHCP lease succeeds while the relay port stays blocked. Check
from the robot first, then from outside:

```bash
ss -ltnp | grep 10000                  # relay listening on 0.0.0.0:10000?
sudo nft list ruleset | head -40       # or: sudo iptables -L INPUT -n
```

If it's a firewall, open the port on the Quest interface, e.g.
`sudo ufw allow in on <wire-iface> to any port 10000 proto tcp`.

**2. The deploy binary's interface is auto-detected, and the fallback is silent.**
`deploy.sh real` looks for an interface holding `192.168.123.x`; if it finds
none, it takes *the first non-loopback interface* with only a yellow warning —
which, now that the Quest wire exists, can easily be `192.168.77.1`. DDS then
goes nowhere while everything looks healthy. **Always read the
`Resolved interface:` line in the deploy banner** and confirm it is the robot LAN
port, not the Quest wire and not `docker0`.

**3. The hand service and the deploy binary must be on the same interface.**
`launch_robot_side.sh` starts `brainco_hand_server` with **no `-n`** unless you
set `ROBOT_IFACE`, so it uses its own default. If that differs from the
interface deploy resolved, `rt/brainco/*` never meets and the arms move with dead
fingers. Set it explicitly:
`ROBOT_IFACE=<192.168.123.x iface> ./scripts/launch_robot_side.sh`.

**4. Both containers need host networking, for different reasons.** The deploy
container needs `--host-net` so DDS sees the real robot LAN (its bridge default
exists for rootless Docker on the lab machine). The relay container needs
`--network-host` so `localhost:5555` reaches the on-board camera server and so
port 10000 is bound on the Quest wire rather than behind Docker NAT. The launcher
passes both; if you start either by hand, don't drop them.

**5. ROS2 and Unitree DDS both live on domain 0.** With host networking, ROS2
discovery would otherwise share the robot LAN with the motor traffic.
`setup_env.sh` sets `ROS_LOCALHOST_ONLY=1`, which confines it — but the Quest
pipeline never needs ROS2, so **`OUTPUT_TYPE=zmq` is the safer choice** and it
also drops the ROS2 dependency from the build.

**6. Nothing binds to localhost-only** — the manager (5556), deploy (5557) and
camera server (5555) all bind `tcp://*`, so the laptop can reach them over the
cable once routing works. If the recorder can't connect, it's the cable, the
static IP, or a firewall — not a bind address.

**7. Stale containers block a restart.** Both containers run under fixed names
(`g1-deploy-dev`, `quest-relay`). If a pane was killed uncleanly you'll get
"name already in use":

```bash
docker rm -f g1-deploy-dev quest-relay
./scripts/launch_robot_side.sh kill      # then relaunch
```

**8. Power on the Quest's ethernet adapter.** Use a USB-C adapter with PD
passthrough. Without it the headset runs the port off its own battery and the
link drops mid-session — which looks like a teleop freeze, not a network fault.

**9. Check the path end to end before trusting it:**

```bash
# [ROBOT] Quest got a lease?
ip neigh show dev <wire-iface>
# [ROBOT] all five ports listening?
ss -ltnp | grep -E '5555|5556|5557|5559|10000'
# [LAPTOP] robot reachable, and the recorder's three sources open?
ping -c3 $ROBOT_IP
for p in 5555 5556 5557; do timeout 2 bash -c "</dev/tcp/$ROBOT_IP/$p" \
    && echo "$p open" || echo "$p CLOSED"; done
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Arms move, fingers dead | `brainco_hand_service` down, or on a different iface/DDS domain than deploy | `systemctl status brainco_hand`; both must be on the robot's `192.168.123.x` iface, domain 0 |
| Never leaves `WAIT_FOR_CONTROL` | manager hasn't sent START | press `s` in the manager pane; check it is connected to the relay |
| Unity won't connect | wired link, relay, or a firewall on port 10000 | see §6.1 — the usual cause is `ipv4.method shared` giving a lease while the firewall drops TCP 10000 |
| Relay image fails to build | missing pyzmq pin (wrong branch) | you are not on `robot-deploy` — the Jetson's Python 3.8 has no cp38 aarch64 wheel for pyzmq ≥ 26 |
| `Missing file` at deploy startup | checkpoints never downloaded | `python download_from_hf.py` (§2c) — `git lfs pull` does not fetch them |
| Recorder hangs at startup | waiting for `robot_config` from deploy | deploy needs `--output-type all`/`zmq`; check `--state-zmq-host` |
| Deploy talks DDS, robot ignores it | wrong interface resolved | check the "Resolved interface" banner is the internal `192.168.123.x` port |
| Slow first start | rebuilding TRT engines from ONNX | expected on new hardware; cached afterwards |
| Thumb curls wrong | `--brainco-thumb-swap` | applied automatically by `deploy.sh real`, never in `sim` |
| Deploy container up, but no DDS to the robot | started without `--host-net` | the launcher passes it; if running by hand, `run-ros2-dev.sh --host-net` |
| `No NVIDIA GPU support detected` from the container | nvidia runtime not configured | `docker info \| grep -i runtime`; on JetPack install `nvidia-container-toolkit` |
| Deploy pane sits at a container shell | the auto-type watcher missed the ready marker | paste the printed `source scripts/setup_env.sh && ./deploy.sh …` yourself |
| Deploy pane waits at `[Y/n]` | expected — this is the safety gate | type `y` once the robot is suspended and the e-stop is in hand |

---

## 8. The other documents

| Doc | Status |
|---|---|
| **this file** | current; the runbook |
| `deployment_setup.md` | current for this topology and updated for the cable — the flag-by-flag manual walkthrough behind the launch scripts |
| `docs/quest_manager_docs.md` | current; the deep dive on the manager (frames, calibration, retargeting, crouch) |
| `deployment_report.md` | **partly stale** — architecture, ports, DDS and BrainCo sections are still accurate and worth reading, but it predates the launch scripts, says `brainco_hand_service` is untracked (it is a submodule now), and says `deploy.sh` has no confirmation prompt (it does; `-y` skips it) |
