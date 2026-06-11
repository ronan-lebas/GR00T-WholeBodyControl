# Quest Manager Thread Server — Documentation

`gear_sonic/scripts/quest_manager_thread_server.py`

This document explains the script end-to-end: what it does, the data it
consumes and produces, the geometry/math behind the pose tracking, and the
operational subtleties (calibration, pause/resume, walking, replay).

---

## 1. Purpose & Role in the System

The script is a **teleoperation manager**. It sits between the Meta Quest 3
headset and the Whole-Body-Control (WBC) deploy policy running on the Unitree
G1 robot. It converts the operator's **3-point VR tracking** (head + both
wrists) plus **hand landmarks** into robot commands and streams them to the
policy.

It is the Quest-based replacement for the older Pico manager
(`pico_manager_thread_server.py`). It targets the **BrainCo hands** variant of
the G1.

### Data flow

```
Quest (Unity) --TCP--> ros_tcp_endpoint --ROS1--> quest_relay (Docker)
    --ZMQ SUB "quest_data" (msgpack, port 5559)--> THIS SCRIPT
    --ZMQ PUB "command"/"planner"/"manager_state" (port 5556)--> deploy policy
```

The script also **subscribes** to a feedback stream (`g1_debug`, port 5557) to
read the robot's measured joint angles for recalibration.

### Frames & conventions

- **Input** (from quest_relay) is in the **ROS FLU frame**: X-forward,
  Y-left, Z-up. Quaternions are **scalar-first** `[w, x, y, z]`.
- **Output** (`planner` topic) is in the **robot ROOT frame**, also
  X-forward, Y-left, Z-up.
- Quaternions are **scalar-first throughout** the script.

---

## 2. Inputs and Outputs

### Input — `quest_data` snapshot (per frame)

| Field | Shape | Meaning |
|---|---|---|
| `head_pos` | `[3]` | head position (FLU) |
| `head_quat` | `[4]` | head orientation, wxyz |
| `left/right_wrist_pos` | `[3]` | wrist position |
| `left/right_wrist_quat` | `[4]` | wrist orientation, wxyz |
| `left/right_landmarks` | `[21][3]` | **MANO-21** hand landmarks |
| `left/right_tracked` | bool | whether that hand is currently tracked |
| `timestamp` | float | sensor timestamp (seconds) |

> A `timestamp <= 0.0` means the relay has not yet received any Quest topic —
> such frames are dropped (not usable for tracking or calibration).

### Output — `planner` topic (per frame, while running)

| Field | Shape | Meaning |
|---|---|---|
| `vr_3pt_position` | `[9]` | 3 rows `[L-wrist, R-wrist, head]`, **key-frame points** (offsets already applied) |
| `vr_3pt_orientation` | `[12]` | scalar-first quaternions for the same 3 rows |
| `left/right_hand_position` | `[7]` | BrainCo: 6 normalized motors `[0=open, 1=closed]` + 1 padding slot |
| `mode` + `movement[3]` + `speed` + `facing[3]` | — | locomotion command |

**Key subtlety on positions:** the wrist rows are *key-frame points*, i.e. the
fixed local offsets (`[0.18, ∓0.025, 0]` for wrists, `[0, 0, 0.35]` above the
torso for the head) are **already applied** and rotated by the live commanded
orientation. The deploy uses the sent values directly (see
`g1_deploy_onnx_ref.cpp` `GatherVR3PointPosition`). The script does this
re-application itself at runtime.

### Output — `command` topic

Start/stop messages for the policy (with `planner=True`).

### Output — `manager_state` topic (every loop)

`stream_mode`, `toggle_data_collection`, `toggle_data_abort`.

---

## 3. Stream / Locomotion Mode Constants

```python
STREAM_MODE_OFF = 0
STREAM_MODE_PLANNER_VR_3PT = 5

LOCOMOTION_IDLE = 0        # stationary; only turns via facing
LOCOMOTION_SLOW_WALK = 1   # translate base along movement_direction
LOCOMOTION_WALK = 2
WIRE_HAND_DOF = 7          # 6 BrainCo motors + 1 pad
```

The stream mode mirrors the deploy state machine. Locomotion modes come from
`localmotion_kplanner.hpp`.

---

## 4. Rotation Helpers

All quaternions are scalar-first. The helpers wrap scipy `Rotation`:

```python
def _rot(quat_wxyz) -> sRot     # safe quat -> Rotation (identity if ~zero norm)
def _quat(rot) -> np.ndarray    # Rotation -> wxyz
def _yaw_of(quat_wxyz) -> float # heading = atan2 of rotated +X axis on XY plane
def _rz(yaw) -> sRot            # rotation about Z
def _wrap(angle) -> float       # wrap to (-pi, pi]
def _slerp(q0, q1, alpha)       # shortest-path normalized lerp (nlerp)
```

`_yaw_of` extracts **heading only** by projecting the rotated forward axis onto
the XY plane — this discards pitch/roll, which is what we want for the
operator's planar facing direction.

`_slerp` is actually an **nlerp** (normalized linear interpolation) with a
sign-flip for shortest path. It's adequate for the short ramps used here
(resume blending, replay prefix blending).

---

## 5. Robot FK Reference — the Calibration Anchor

### `RobotRestReference`

Loads the G1 BrainCo robot model and provides forward-kinematics (FK) of the
wrist links and torso. This is the **robot-side anchor** that the operator's
rest pose maps onto during calibration.

#### The hand-convention rotation `N`

The Quest publishes wrist orientation in the **physical hand convention**
(X = fingers, Z = back of hand), verified against the MANO landmarks in
recorded data. But the **G1 wrist link frame** is rolled ~90° about the
forearm axis relative to that. If you compare the two raw frames directly,
every wrist orientation comes out 90° off (a palms-down operator shows up as
the robot's palms-inward rest pose).

`_compute_hand_convention(side)` computes a **constant local rotation `N`**
from the wrist *link* frame to the *physical* hand frame, derived from the
BrainCo finger FK geometry:

```python
fingers = mid - wrist_pos            # wrist -> middle finger = +X (fingers)
across  = idx - pky                  # index -> pinky spans knuckles
back    = cross(fingers, across)     # (sign flips per side) = back of hand
y       = cross(back, fingers)       # orthonormalize
back    = cross(fingers, y)
phys    = R[[fingers, y, back]]      # physical hand frame in link coords
return wrist_rot.inv() * phys        # N: link -> physical
```

The left/right URDFs capitalize the proximal link names differently
(`proximal_Link` vs `proximal_link`) — handled per side.

`N` is only used directly when **synthesizing replay rest frames** (Section
9). In live tracking the convention cancels out automatically via the
world-frame delta (Section 6).

#### `compute(body_q_29)` — FK key-frame poses

Returns the wrist **link** poses (no key-frame offset applied) and the torso
position, in the root frame:

```python
{"left": (link_pos, link_rot), "right": (...), "torso_pos": (3,)}
```

If `body_q_29` is given (measured joints), FK uses that configuration;
otherwise it uses the model's **default rest pose**. The offsets are
re-applied later at runtime with the *live* orientation, which is why FK
returns the bare link origins here.

### `RobotFeedback`

Subscribes to the deploy `g1_debug` ZMQ stream and unpacks
`body_q_measured` (the robot's actually-measured 29-DOF joints). Used at
**recalibration** so the calibration anchor reflects the robot's *real* current
pose rather than the idealized rest pose.

---

## 6. The Core: `QuestThreePointTracker`

This is the heart of the script. It maps Quest head/wrist poses into
robot-root-frame VR 3-point targets, with calibration that makes tracking
robust to the operator walking, leaning, and turning.

### 6.1 Calibration

Performed once with the operator in the **rest pose** mirroring the robot's
reference pose. Captures:

- **`R0 = Rz(head_yaw)`** — the operator's heading frame at calibration. All
  runtime quantities are expressed in this fixed frame, taken to coincide with
  the robot root frame.
- **`v_cal = R0⁻¹ (p_wrist − p_head)`** — the head→wrist offset vector. Using
  the **head-relative delta** (not absolute wrist position) makes tracking
  invariant to the operator walking or leaning.
- **`W_cal`** — operator wrist orientation at rest. Captures the unknown,
  fixed Quest-hand-frame vs robot-wrist-frame convention offset.
- **`(link_ref, R_ref)`** — robot wrist LINK pose from FK; the robot pose the
  operator's rest pose maps onto.

```python
def calibrate(self, frame, body_q_29=None):
    fk = self._robot_ref.compute(body_q_29)
    self._yaw_cal = _yaw_of(frame["head_quat"])
    self._r0 = _rz(self._yaw_cal); self._r0_inv = self._r0.inv()
    head_pos = frame["head_pos"]
    for side in _SIDES:
        wrist_pos = frame[f"{side}_wrist_pos"]
        self._v_cal[side]     = self._r0_inv.apply(wrist_pos - head_pos)
        self._w_cal_inv[side] = _rot(frame[f"{side}_wrist_quat"]).inv()
        self._link_ref[side], self._rot_ref[side] = fk[side]
    self._torso_pos = fk["torso_pos"]
    # reset head-velocity tracking ...
```

### 6.2 Runtime mapping (per wrist)

With `Rh(t) = Rz(head_yaw(t))` the **current** heading frame:

```
v(t)    = Rh(t)⁻¹ (p_wrist(t) − p_head(t))
link(t) = link_ref + pos_scale · (v(t) − v_cal)
R(t)    = Rh(t)⁻¹ · W(t) · W_cal⁻¹ · R0 · R_ref      (world-frame delta)
sent(t) = link(t) + R(t) · key_frame_offset
```

In code:

```python
yaw_rel = _wrap(_yaw_of(frame["head_quat"]) - self._yaw_cal)
rh_inv  = _rz(self._yaw_cal + yaw_rel).inv()          # current heading frame
...
v    = rh_inv.apply(wrist_pos - head_pos)
link = self._link_ref[side] + self.pos_scale * (v - self._v_cal[side])
w    = _rot(frame[f"{side}_wrist_quat"])
r_cmd = rh_inv * w * self._w_cal_inv[side] * self._r0 * self._rot_ref[side]
pos[i]  = link + r_cmd.apply(self._WRIST_OFFSET[side])
quat[i] = _quat(r_cmd)
```

#### Why the **current** heading frame `Rh(t)` (not fixed `R0`)?

Using the live heading makes the targets **body-relative**. When the operator
turns their whole body about Z, both `v(t)` and `R(t)` are unchanged, so the
arms do **not** chase the rotation. The rotation is instead returned separately
as `yaw_rel` and sent as the planner **`facing`** command, which turns the
robot's whole body to follow the operator.

#### Why the **world-frame delta** for orientation?

`R(t) = Rh⁻¹ · W · W_cal⁻¹ · R0 · R_ref` makes the orientation independent of
the Quest hand-frame convention. Any fixed offset `C` in `W = W_physical · C`
**cancels** in the `W · W_cal⁻¹` product. This is why `N` does not need to be
applied in the live path.

#### Head row

```python
pos[2]  = self._torso_pos + self._HEAD_OFFSET   # torso + [0,0,0.35]
quat[2] = [1, 0, 0, 0]                           # identity
```

The torso target stays aligned with the root (which itself rotates via
`facing`). Position comes from the fixed kinematic chain.

The wrist key-frame offsets:

```python
_WRIST_OFFSET = {"left": G1_KEY_FRAME_OFFSETS["left_wrist"],
                 "right": G1_KEY_FRAME_OFFSETS["right_wrist"]}
_HEAD_OFFSET  = G1_KEY_FRAME_OFFSETS["torso"]
```

### 6.3 Head-velocity estimation (drives walking)

`compute()` also returns the operator's **planar head velocity** in the fixed
`R0` frame, used to drive base locomotion.

```python
_VEL_TAU = 0.25         # low-pass time constant (s)
_VEL_STALE_SEC = 0.3    # decay-to-zero window when stream freezes
```

`_update_head_velocity(head_pos, ts)`:

- Differences position over **sensor timestamps** (`ts`), not loop time. So a
  repeated/sample-and-held frame contributes **no spurious motion**.
- If sensor time jumps **backwards** (replay looped or stepped back), it
  re-anchors on the new frame instead of differencing across the
  discontinuity.
- Applies a timestamp-aware exponential low-pass: `alpha = dt / (VEL_TAU + dt)`.
- Forces `raw[2] = 0.0` (planar only).
- If no fresh frame arrives within `_VEL_STALE_SEC`, the velocity **decays to
  zero** so the robot doesn't keep walking on a dropped stream.

```python
if dt > 1e-4:
    raw = self._r0_inv.apply(head_pos - self._prev_head_pos) / dt
    raw[2] = 0.0
    alpha = dt / (self._VEL_TAU + dt)
    self._head_vel = (1.0 - alpha) * self._head_vel + alpha * raw
    ...
elif now - self._last_vel_wall > self._VEL_STALE_SEC:
    self._head_vel = np.zeros(3)
```

### 6.4 `compute()` return signature

```python
(vr_position (9,), vr_orientation (12,), yaw_rel (float), head_vel (3,))
```

- `vr_position`/`vr_orientation`: body-relative targets in the root frame.
- `yaw_rel`: operator heading change since calibration → send as `facing`.
- `head_vel`: planar head velocity (m/s, z=0) in R0 frame → drives walking.

---

## 7. Finger Retargeting (MANO-21 → 6 BrainCo motors)

### MANO-21 → XR-25 conversion

The optimization-based retargeter expects the **XR-25** layout (OpenXR without
palm): index 0 is the wrist, then per finger `[metacarpal, proximal,
intermediate, distal, tip]` (thumb has 4 joints, others 5).

MANO-21 (MediaPipe) has **no metacarpal points**, so those are **synthesized**
as the wrist→MCP midpoint:

```python
def mano21_to_xr25(landmarks21):
    xr = np.zeros((25, 3))
    xr[0] = lm[0]
    for xr_i, mano_i in _XR25_FROM_MANO:       # direct copies
        xr[xr_i] = lm[mano_i]
    for xr_i, mano_mcp in _XR25_METACARPALS:   # synthesized midpoints
        xr[xr_i] = 0.5 * (lm[0] + lm[mano_mcp])
    return xr
```

### `FingerRetargeting`

Two backends, selected at init:

1. **`BrainCoRetargeter`** (optimization-based) — preferred when importable.
   `canonicalize(xr, side)` then `retarget_left/right(canon)`.
2. **`np_retargeting`** (pure-numpy angle-based) — fallback. Reads per-joint
   angles and normalizes each by its joint limit.

Forced to numpy with `--np-retarget`. If neither import is available, raises
`ImportError`.

Output is always a **7-element wire vector**: 6 motors clipped to `[0, 1]` plus
a `0.0` pad:

```python
return [float(np.clip(m, 0.0, 1.0)) for m in motors] + [0.0]
```

Wire order (numpy backend): `thumb_metacarpal, thumb_proximal, index_proximal,
middle_proximal, ring_proximal, pinky_proximal` — matches BrainCo firmware /
mock streamer convention.

---

## 8. Data Sources

The manager is agnostic to where frames come from. Both sources expose
`get_frame() -> dict | None`, an `is_replay` flag, and `close()`.

### `LiveQuestSource`

A background ZMQ **SUB** thread holding the latest `quest_data` snapshot:

- Connects to `tcp://host:port`, subscribes to topic `quest_data`,
  `RCVTIMEO = 100ms`.
- `_run()` loops `recv_multipart()`, unpacks msgpack, normalizes via
  `_frame_from_snapshot`, **drops frames with `timestamp <= 0.0`**, and stores
  the latest under a lock.
- `get_frame()` returns the latest snapshot (or `None` before the first).

### `ReplaySource`

Replays a recorded NPZ trajectory (from `record_quest_data.py`). Key features:

- **Synthetic rest-pose prefix** (Section 9) so calibration on frame 0 is
  always valid.
- **Time-anchored playback**: wall clock vs recorded timestamps; medians the
  inter-frame `dt` (fallback 1/50 s).
- **Loops** at the end (`media_time = elapsed % duration`).
- Supports **pause** (`k`) and **single-frame stepping** (arrows).

Playback index resolution:

```python
def _current_media_time(self):
    return self._media_time if self._paused else \
           (time.monotonic() - self._anchor) % self._duration

def _current_index(self):
    idx = searchsorted(self._ts, self._current_media_time(), "right") - 1
    return clip(idx, 0, self._n - 1)
```

---

## 9. The Replay Rest-Pose Prefix

Live teleop relies on the operator physically assuming the rest pose during a
countdown before calibration. A recording has no such countdown, so calibrating
on the recording's frame 0 would be wrong.

`_build_rest_prefix` **synthesizes** frames that place a *virtual* operator
exactly in the pose calibration expects, then blends into the real recording:

1. Compute FK rest poses; define `head_ref = torso_pos + HEAD_OFFSET`.
2. Build a **rest frame** whose head matches the recording's first head pose,
   and whose wrists sit where the robot's rest wrist links sit relative to the
   head, lifted into the Quest world by the head heading `r0`:

   ```python
   rest[f"{side}_wrist_pos"]  = head_pos0 + r0.apply(link_pos - head_ref)
   rest[f"{side}_wrist_quat"] = _quat(r0 * link_rot * hand_convention(side))
   ```

   The synthetic wrist quaternion **must** be what the Quest would publish:
   the link orientation expressed in the **physical hand convention `N`**
   (because the Quest wrist TF is physical, not link). This is the one place
   `N` is applied directly.

3. **Hold** the rest frame for `rest_hold_sec`, then **interpolate** to the
   recording's first frame over `rest_interp_sec` (lerp for positions, `_slerp`
   for quaternions, landmarks held constant).
4. Tracked flags are `False` during the prefix (no finger retargeting).

Calibrating on this synthetic frame 0 maps it **exactly** onto FK rest.

---

## 10. Keyboard Input — `KeyboardListener`

A cbreak (`tty.setcbreak`) single-key reader on stdin, running in a background
thread, pushing keys into a `queue.Queue`. Arrow-key escape sequences
(`ESC [ C` / `ESC [ D`) are decoded to `"RIGHT"` / `"LEFT"`. If stdin is not a
TTY, controls are disabled with a warning. Terminal settings are restored on
exit (`__exit__`).

### Key bindings

| Key | Action |
|---|---|
| `s` | start policy + enter VR_3PT mode (countdown on live Quest) |
| `r` | recalibrate (countdown; uses measured robot joints as FK ref) |
| `f` | toggle finger retargeting |
| `p` | pause / resume teleop (freeze robot, smooth resume) |
| `c` | toggle data collection |
| `x` | toggle data abort |
| `q` | stop policy and exit |
| `k` | (replay) pause / resume playback |
| `←` / `→` | (replay) step one frame back / forward |

---

## 11. The Manager — `QuestManager`

Owns all components: `RobotRestReference`, `QuestThreePointTracker`,
`FingerRetargeting`, `RobotFeedback`, the data source, and the ZMQ **PUB**
socket (binds `tcp://*:port`, with a 0.3 s sleep so subscribers connect before
the first messages).

### 11.1 State

- `stream_mode`, `finger_tracking`, `teleop_paused`
- Walking: `walk_mode`, `_walking` (hysteresis flag)
- `resume_ramp_start`, `resume_rebase_pending`
- `pending_calib = (deadline, kind)`, countdown print state
- Frozen targets for pause: `frozen_pos/quat/yaw/hands`
- `last_hands` (held while a hand is untracked)
- `yaw_offset`, `last_facing_yaw` — heading rebasing (see 11.4)

### 11.2 Calibration triggering

`_arm_calibration(kind)`:

- **Replay**: no countdown — the synthetic prefix *is* the rest pose. On
  `"start"` it restarts the source and calibrates on frame 0 immediately.
- **Live**: arms a countdown (`calib_delay_sec`, default 3 s) and prints a
  decreasing counter.

`_do_calibration(kind, frame)`:

- For `"recalib"`, fetches `measured_body_q()` so FK uses the robot's real
  joints; warns and falls back to the default rest pose if feedback is missing.
- Calls `tracker.calibrate(...)`.
- On `"start"`: zeroes `yaw_offset`/`last_facing_yaw`, sends the policy
  **START** command, and switches to `STREAM_MODE_PLANNER_VR_3PT`.
- On `"recalib"`: sets `yaw_offset = _wrap(-last_facing_yaw)` so the commanded
  facing stays **continuous** (the robot doesn't snap back to its start
  heading even though `yaw_rel` was just zeroed).

### 11.3 Hands — `_compute_hands`

If finger tracking is off, returns `{left: None, right: None}`. Otherwise, per
side, retargets only when that hand is **tracked**, and otherwise **holds the
last command** (`last_hands`). Retargeting exceptions are caught and logged so
one bad frame doesn't kill the loop.

### 11.4 Pause / resume — `_apply_pause_resume`

```python
facing_yaw = _wrap(yaw_rel - self.yaw_offset)
```

**Paused:** the first paused frame captures `frozen_pos/quat/yaw/hands`;
subsequent frames return the frozen values → robot is frozen at the last pose.

**Resuming:** ramps from frozen to live over `resume_ramp_sec`:

- On the first resume frame, **rebases heading**: `yaw_offset = _wrap(yaw_rel −
  frozen_yaw)` so facing continues from the frozen heading (the operator may
  have turned while paused) instead of jumping.
- Blends position linearly and orientation via `_slerp` per row with
  `alpha = (now − start) / resume_ramp_sec`; when `alpha >= 1` the ramp ends.

This `yaw_offset` mechanism (used by both recalibration and resume) is what
guarantees the commanded `facing` never jumps discontinuously.

### 11.5 Walking — `_walk_command`

Maps the operator's head planar velocity (R0 frame) to a planner movement:

```python
if self.args.disable_walk or self.teleop_paused:
    self._walking = False
    return LOCOMOTION_IDLE, [0,0,0], -1.0

speed = hypot(head_vel[0], head_vel[1])
# Hysteresis: start above deadband, keep going until well below it
if self._walking:
    self._walking = speed >= 0.5 * walk_deadband
else:
    self._walking = speed >= walk_deadband
if not self._walking:
    return LOCOMOTION_IDLE, [0,0,0], -1.0

direction  = head_vel[:2] / max(speed, 1e-6)
cmd_speed  = clip(speed * walk_speed_scale, walk_min_speed, walk_max_speed)
return self.walk_mode, [direction[0], direction[1], 0.0], cmd_speed
```

Subtleties:

- **Hysteresis** around the deadband (start at `walk_deadband`, stop at
  `0.5 * walk_deadband`) prevents flapping between IDLE and walk.
- Disabled walking / pause / below-deadband all yield IDLE with a zero
  movement vector — **exactly the pre-walk behavior** (robot still turns in
  place via `facing`).
- Note: the **head ROW** of `vr_position` stays fixed; walking is a separate
  planner command, *not* a moving torso target.

### 11.6 Main loop — `run()`

Runs at `target_fps` (default 50 Hz). Each iteration:

1. Drain the keyboard queue; handle keys (may set quit / data-collection /
   data-abort toggles).
2. `frame = source.get_frame()`; update calibration countdown.
3. If in VR_3PT mode and a frame exists:
   - `tracker.compute(frame)` → pos, quat, yaw_rel, head_vel
   - `_compute_hands(frame)`
   - `_apply_pause_resume(...)` → possibly frozen/blended pos, quat, facing_yaw
   - `_walk_command(head_vel)` → mode, movement, speed
   - Publish the **planner** message with `facing = [cos(facing_yaw),
     sin(facing_yaw), 0]`, `height = -1.0`, hand positions, and the VR 3-point
     position/orientation.
4. Always publish the **manager_state** message (stream mode + toggles).
5. Every 5 s, print a status line (state, quest data ok/waiting, planner
   msg/s, fingers on/off, walk state).
6. Sleep to maintain the target period.

On exit (clean or `KeyboardInterrupt`), sends the policy **STOP** command and
closes sockets.

---

## 12. Command-Line Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--port` | 5556 | ZMQ PUB port (deploy subscribes) |
| `--relay-host` / `--relay-port` | localhost / 5559 | quest_relay source |
| `--feedback-host` / `--feedback-port` | localhost / 5557 | g1_debug feedback |
| `--target-fps` | 50 | planner stream rate |
| `--pos-scale` | 1.0 | scale on operator arm reach (orientation unaffected) |
| `--calib-delay-sec` | 3.0 | countdown after `s`/`r` before capturing the calibration frame (live only) |
| `--resume-ramp-sec` | 1.0 | ease-in duration from frozen to live pose on resume |
| `--disable-walk` | off | robot only turns in place; never translates |
| `--walk-mode` | slow | `slow` (SLOW_WALK) or `walk` (WALK) |
| `--walk-deadband` | 0.08 | head speed (m/s) above which walking starts |
| `--walk-speed-scale` | 1.0 | scale from operator head speed to robot walk speed |
| `--walk-min-speed` / `--walk-max-speed` | 0.2 / 0.8 | clamp on commanded walk speed |
| `--np-retarget` | off | force the pure-numpy finger retargeter |
| `--replay` | None | NPZ trajectory to replay instead of live Quest |
| `--rest-hold-sec` / `--rest-interp-sec` | 1.0 / 1.0 | (replay) rest-pose hold + blend durations |

### Typical invocations

```bash
# Live Quest (relay container running)
python quest_manager_thread_server.py

# Turn-in-place only, no base translation
python quest_manager_thread_server.py --disable-walk

# Replay a recorded trajectory
python quest_manager_thread_server.py --replay data/quest/traj_xxx.npz
```

---

## 13. Summary of the Clever Bits

1. **Head-relative wrist deltas** (`v = R⁻¹(p_wrist − p_head)`) make tracking
   invariant to the operator walking/leaning.
2. **Current heading frame `Rh(t)`** for arm targets + separate `facing`
   command means body rotation turns the robot without the arms chasing it.
3. **World-frame orientation delta** (`Rh⁻¹ W W_cal⁻¹ R0 R_ref`) automatically
   cancels the unknown Quest-vs-link hand-frame convention.
4. **Timestamp-based head velocity** with staleness decay gives robust,
   loop-rate-independent walking that doesn't drift on dropped/held frames.
5. **`yaw_offset` rebasing** keeps `facing` continuous across recalibration and
   pause/resume.
6. **Synthetic rest-pose prefix** lets replay calibrate on frame 0 exactly as
   if the operator had assumed the rest pose live.
7. **Walking hysteresis** + IDLE fallback that reproduces the exact pre-walk
   behavior when disabled.
8. **Key-frame offsets applied here** with live orientation, since the deploy
   consumes the sent values directly.
