# Optimization-based reference generation for chair manipulation

Teleoperating the chair through the Quest is hard: the operator has to solve reach, balance and
a two-hand grasp at the same time, and most attempts end with the chair knocked over rather
than reoriented. This is the alternative path — *generate* the trajectory offline by
optimization, then train an RL controller to track it, so the operator (or a task policy) only
has to say where the chair should end up.

    python gear_sonic/scripts/generate_chair_trajectory.py --rotate-deg 45

produces a 50 Hz reference — robot base pose + 29 body joints + 12 finger joints + the chair's
6-DOF pose + a contact schedule — that starts from the stack's own scene (chair at
`CHAIR_POS_X = 0.55` in front of a static-base G1 with BrainCo hands) and ends with the chair
where it was asked to be.

Everything lives in `gear_sonic/trajopt/`; the entry point is
`gear_sonic/scripts/generate_chair_trajectory.py`. Runs in `.venv_sim` (mujoco + scipy; no new
dependencies).

## Method

The structure follows **FARO** (Ciebielski, Omar, Johnson, Khadiv, *FARO: Feasibility-Aware
Robot Motion Optimization*, arXiv:2607.18362), whose central idea is that multi-contact motion
planning is dominated by the cost of *discovering which contact sequence works*, and that this
is best done by filtering candidate sequences through progressively more expensive feasibility
tests rather than by throwing a full trajectory optimization at each one.

| FARO | Here | Where |
|---|---|---|
| Contact-mode sequence `C = (c_0…c_{K-1})`, §II-A | patch-per-hand + chair-on-floor flag per stage; K+1 keyframes | `plans.py` |
| Patch-to-patch contact (7a–7b), no-slip (8) | palm pad on a rectangular patch, normal anti-parallel, rotation about it free. The in-patch location is a *decision variable shared over a contact episode* — no-slip holds by construction, not as a constraint | `chair.py`, `kso.py` |
| Collision via SDF/GJK (9) | `mj_geomDistance` on a curated geom-pair list, hinge residuals, bounding-sphere prefilter | `scene.py` |
| Limits (13) | position/velocity/torque limits parsed from the G1 URDF | `scene.py` |
| Mode / edge feasibility IK (14) | single-configuration solve per mode and per transition, cached by (mode identity, chair pose) | `kso.py` |
| Kinematic sequence optimization (15) | all keyframes at once, coupled by shared contact locations and a smoothness term | `kso.py` |
| Full trajectory optimization (17) | **replaced** — see below | `trajectory.py`, `dynamics.py` |
| Feasibility-guided tree search (Alg. 1) | small ordered plan library + the same `M → E → KSO → dynamics` cascade, first survivor wins | `kso.evaluate_plan`, `plans.candidate_plans` |

Each problem is one sparse nonlinear least-squares solve (`scipy.optimize.least_squares`, TRF,
analytic sparsity pattern, finite-difference Jacobian). Hard constraints enter as bounds
(joint limits, in-patch coordinates, object slack) or as heavily weighted residuals;
inequalities (collision, balance) as one-sided hinges. **Feasibility is decided on the
unweighted violations**, so weights only shape convergence and never the verdict — and rather
than hand-tuning weights until every constraint happens to land inside tolerance, `solve()`
runs a penalty continuation: solve, multiply the weights of whatever is still violated,
re-solve warm-started.

### What replaces FARO's trajectory optimization

FARO closes its hierarchy with a collocation NLP over whole-body and object dynamics, contact
wrenches and stage timings (casadi + acados/Hippo). That stack is not in this repo, and the
reference does not need it: the RL controller re-derives contact forces during training. What
the reference *does* need is to not be physically absurd. So:

1. **Path before time** (`trajectory.py`). Each stage is densified into knots by solving the
   same whole-body IK with the chair pinned to the interpolated object motion and the contact
   locations frozen at the values KSO chose. Knots are solved *jointly* per stage, not one at a
   time — an independent solve per knot lands in different parts of the arm's null space and
   the resulting path zig-zags, which reads downstream as metre-per-second-squared CoM
   accelerations. Time enters only afterwards, through per-stage durations, so retiming is free.
2. **Kinodynamic checks** (`dynamics.py`): the wrench the hands must apply to move the chair as
   commanded, split over the active contacts and tested against friction cone, torsional
   friction and centre-of-pressure bounds; the ZMP of the robot (carrying whatever the hands
   hold, taking the reaction of whatever they push) against the foot polygon; joint velocities
   and inverse-dynamics torques against the URDF limits.
3. **Retiming as the repair** (FARO's `T̄_s`): every violation that scales with acceleration is
   fixed by stretching stage durations, which the generator does automatically. Violations that
   do *not* scale — a grasp that must resist more moment than the contact can hold, a posture
   whose gravity torque exceeds the motors — stop the loop and reject the plan, and the search
   moves to the next candidate. That is FARO's TO filter in the role it actually plays.

### Plan library

`--plan auto` (default) tries, best-first: **slide** (pinch two opposing faces, swivel the
chair on the floor — the legs keep carrying the weight, so nothing has to be balanced),
**push** (both palms on one backrest face, floor-supported), **lift** (pinch and carry, per
patch pair, ranked by how laterally the robot can squeeze and how far it must reach),
**rail_lift** (both palms over the backrest top rail), and for large displacements **liftx2**
— a regrasp: set the chair down, let go, take hold of the face that is *now* lateral, continue.
That last one exists because with both feet planted the wrists run out of range past roughly
half a turn, so a large rotation is only reachable as a sequence of contact modes.

## Results

Chair: the hardcoded SANDSBERG-like proxy (seat 0.40 × 0.40 m at 0.45 m, 0.86 m tall, 0.2 kg,
matching `run_sim_loop.CHAIR_MASS`); a staged asset is used for its bounding box and mesh when
`data/objects/chair` exists. All runs on one CPU, `.venv_sim`.

| Task | Plan chosen | Wall time | Duration | ZMP margin | Contact drift |
|---|---|---|---|---|---|
| rotate 45° in place | `slide[back_py\|back_ny]` | 79 s | 11.6 s | −0.8 cm | 2.6 mm |
| rotate 20° + 15 cm left | `slide[back_py\|back_ny]` | not timed | 5.9 s | +3.9 cm | 0.3 mm |
| rotate 60° in place | `rail_lift` | 206 s | 11.1 s | −0.8 cm | 3.2 mm |
| rotate 90° in place | *none — rejected* | not timed | — | — | — |

Wall time is single-CPU and is dominated by the plans that get *rejected*: the 60° run pays for
six failed candidates before `rail_lift` succeeds, while 45° finds its plan third. It is
essentially all finite-difference Jacobian evaluation.

"Contact drift" is the largest distance a palm moves off its patch surface anywhere in the
sampled trajectory (the keyframes and knots are exact; this is what interpolation between them
costs). The 45° and 60° results sit right at the ZMP tolerance because the G1's feet are 17 cm
long and the chair stands half a metre away, which puts the CoM near the toe edge for the whole
reach; a centimetre of ZMP excursion is inside what a tracking controller absorbs, and the
generator will not accept more (see `dynamics.ZMP_TOLERANCE`).

The 90° rejection is the honest answer, not a solver failure: every candidate is turned down by
a *named* filter — the palm would have to pull on a push patch, or the grasp cannot be held
through mid-rotation, or the CoM leaves the foot polygon. With both feet planted and the chair
at 0.55 m, a quarter turn is outside the robot's envelope. `report.json` records which filter
rejected what, so this is diagnosable rather than mysterious.

## Outputs

Written to `data/generated_trajectories/<name>/`:

- **`trajectory.npz`** — `t`, full-scene MuJoCo `qpos` (robot + chair, directly replayable),
  `whole_q` (51-DOF Pinocchio layout, the same thing the recorder stores as
  `observation.state`), `dof29`, `hand_dof`, the chair pose in both the canonical proxy frame
  and the staged asset's own frame (the latter is what the sim's `object` free joint takes),
  the per-frame contact flags and stage indices.
- **`motion_lib.pkl`** — SONIC training format (`root_trans_offset`, `root_rot` (xyzw), `dof`
  in MuJoCo joint order, `pose_aa`, `fps`), plus `object_trans` / `object_rot` / `contact` /
  `hand_dof` as extra keys, so an object-aware tracking reward has everything in one file.
- **`report.json`** — every candidate plan, which filter rejected it and why, and all
  feasibility margins of the accepted trajectory.
- **`scene.xml`** — the exact MuJoCo scene it was generated against, loadable standalone.

`--replay` plays the result in the MuJoCo viewer (space = pause, arrows = step).

## Known limitations

- **Feet are planted throughout**, matching the stack's `--static-base` teleop. This is the
  binding constraint on almost everything: it is what makes 90° infeasible and what keeps the
  CoM at the polygon edge. Adding a stepping mode (feet as contact interfaces in the mode
  language, which the representation already allows) would open the envelope the most of any
  single change.
- **Balance is checked as a CoM/ZMP condition on the *robot alone***. A robot can legitimately
  lean on what it pushes, but only if that object carries the load — and this chair weighs
  0.2 kg, so the conservative choice is taken. Raising `--chair-mass` does not currently unlock
  leaning; the polygon would have to be extended through the hand contacts.
- **Grasps are modelled as a wrapped grip** (`dynamics.GRIP_FORCE`, `GRASP_LEVER`) rather than
  as a finger-level force closure. The finger trajectory itself is a scripted open/closed ramp,
  not optimized — this is exactly the contact detail the RL controller is expected to supply.
- **Angular-momentum rate is neglected** in the ZMP check, and the torque check splits the
  ground reaction evenly between the feet. Both are magnitude checks, not certificates.
- **The chair proxy is a box abstraction** (seat plate, backrest, four legs) sized from the
  staged asset's bounding box. Contact patches are defined on the proxy, so a chair shaped very
  differently from the SANDSBERG proportions needs its `--seat-height` (or the `ChairSpec`
  defaults) adjusted.
- **9 knots per stage is a sweet spot, not a floor.** More knots make the coupled stage solve
  harder to converge within its iteration budget and the path gets *worse*; `--knots` is
  exposed but 12 already degrades.

## Next steps, in order of expected value

1. Stepping: add the feet to the contact-mode language and let a plan reposition the stance.
   This is what unlocks large rotations and restores real balance margin.
2. Train the tracking controller on a generated reference and measure what actually survives —
   the whole point is that the RL policy absorbs contact detail, and that assumption is
   currently untested.
3. Replace the plan library with FARO's actual tree search over mode sequences (cost-based UCT
   with progressive widening) once stepping makes the search space genuinely large. The filter
   cascade and cache it needs are already here.
4. Analytic derivatives (pinocchio, or casadi if it is ever added to the sim env). The
   finite-difference Jacobian is the entire runtime cost; everything else is milliseconds.
