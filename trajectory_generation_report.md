# Optimization-based reference generation for chair manipulation

Teleoperating the chair through the Quest is hard: the operator has to solve reach, balance and
a two-hand grasp at the same time, and most attempts end with the chair knocked over rather
than reoriented. This is the alternative path — *generate* the trajectory offline by
optimization, then train an RL controller to track it, so the operator (or a task policy) only
has to say where the chair should end up.

    python gear_sonic/scripts/generate_chair_trajectory.py --rotate-deg 45

produces a 50 Hz reference — robot base pose + 29 body joints + 12 finger joints + the chair's
6-DOF pose + a contact schedule for the hands *and the feet* — that starts from the stack's own
scene (chair at `CHAIR_POS_X = 0.55` in front of a G1 with BrainCo hands, standing in its spawn
stance) and ends with the chair where it was asked to be. The robot walks to the chair if it
has to, and repositions its stance mid-task if one stance cannot cover the whole motion.

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
| Contact-mode sequence `C = (c_0…c_{K-1})`, §II-A | patch-per-hand + planted-or-swinging per foot + chair-on-floor flag, per stage; K+1 keyframes | `plans.py` |
| Patch-to-patch contact (7a–7b), no-slip (8) | palm pad on a rectangular patch, normal anti-parallel, bearing about the normal fixed. All three — `(u, v, theta)` — are *decision variables shared over a contact episode*, so no-slip and no-twist both hold by construction rather than as constraints | `chair.py`, `kso.py` |
| — which face of the hand | the palm pad frame is derived from the model: the fingertips' *flexion displacement* gives the outward normal, since the fingers curl toward the palm. Taking it from the thumb instead reads out the hand's lateral axis on the BrainCo hand — whose thumb rests roughly in the plane of the palm — and presses the back and edge of the hand onto the chair | `scene._palm_frame` |
| — which way round the hand | for a grasp patch, the hand's finger-curl axis must lie within 25° of the patch's long side, so the fingers close *across* the thing they hold rather than along it | `kso` `grasp_align` |
| Self-collision | every left-arm link against every right-arm link, and the arm against the chest and pelvis — enumerated, not hand-picked | `scene.SELF_COLLISION_BODY_PAIRS` |
| Feet as contact interfaces | a planted foot owns one `(x, y, yaw)` placement for a whole stance episode — the same construction as a hand's patch point, so it cannot slide either. A step is the boundary between two episodes of one foot | `kso.StanceEpisode` |
| Collision via SDF/GJK (9) | `mj_geomDistance` on an enumerated geom-pair list, hinge residuals, bounding-sphere prefilter, and a guard against the solver's spurious zeros (below) | `scene.py` |
| Limits (13) | position/velocity/torque limits parsed from the G1 URDF | `scene.py` |
| Mode / edge feasibility IK (14) | single-configuration solve per mode and per transition, cached by (mode identity, chair pose, whether the stance is free) | `kso.py` |
| Kinematic sequence optimization (15) | all keyframes at once, coupled by shared contact locations, shared stance placements and a smoothness term | `kso.py` |
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

### What a contact actually pins

A contact used to fix five of the palm's six degrees of freedom: the pad on a point of the
patch, and the palm normal against the patch normal. The sixth — the bearing about that
normal — was left free, and it was not a harmless relaxation. Over the 90° task the palm
rotated **108° relative to the chair** inside a single grasp: the hand stayed put in the
world while the chair turned underneath it. Nothing about that is a grip. It is also
invisible in every margin the generator reported, because the palm never left the patch.

The fix is the same construction the in-patch location already used. A contact episode owns
`(u, v, theta)`, all three shared by every frame it spans, and the residual pins the hand's
own finger-curl axis to the bearing `theta` marks out on the patch. A hand therefore cannot
slide *or* rotate on the object while it holds it — structurally, not as something the
solver is asked to trade off.

`theta` is then bounded for grasp patches, because not every bearing can be closed on: the
curl axis has to lie along the patch's *long* side so the fingers wrap the short one. The
patch's long side is the axis of the feature being held, which is what makes this the right
rule for both a backrest edge (curl axis vertical, fingers wrap the 3.5 cm thickness) and
the top rail (curl axis along the rail, fingers wrap over it).

This constrains the plan library without gutting it: the 45° task still chooses the same
backrest-edge pinch it always did, now holding it with a fixed bearing instead of letting it
spin. It *is* strictly narrower, though — `theta` has to be found alongside everything else,
and some grasps that used to pass do not.

One caution, learned the hard way here. Between adding the twist constraint and fixing the
`mj_geomDistance` artifact described above, the backrest pinch was being rejected for
collision, and the obvious story — "holding a vertical edge with a vertical curl axis puts
the hips into the seat" — was wrong. It was the artifact. When a new constraint makes plans
start failing, the tempting reading is that the constraint has revealed something; check the
mechanism before believing it.

### Arms are checked against each other

The self-collision list used to name twelve body pairs by hand, and the two arms appeared in
it only as `left_base_link / right_base_link`. A two-hand task puts both arms in the same
small volume in front of the chest, so that was not enough: in the 90° trajectory the left
palm ended up **25 mm inside the right wrist**, across 222 of 1955 frames. The list is now
enumerated — every left-arm link against every right-arm link, from the shoulder yaw joint
down, plus each arm against the chest and pelvis — which is 60 body pairs and 90 geom pairs.
The shoulder yaw link is deliberately excluded from the *chest* pairs: it sits 3 cm from the
torso in the nominal stance, inside `COLLISION_MARGIN`, so checking it would report a
permanent violation of a clearance the robot does not physically have.

### `mj_geomDistance` invents contacts

Worth stating on its own, because it silently corrupted the collision term for a long time
and its signature is easy to mistake for a real result. For mesh pairs the convex solver
sometimes returns **exactly 0** for geoms that are provably centimetres apart. It shows up
downstream as a collision residual of exactly `COLLISION_MARGIN` — a plan rejected with
`collision=0.035` is very often this and not a real interpenetration.

The failure only goes one way (0 instead of a real, larger value) and the query radius at
which it starts depends on the *pose*, not just the pair, so it cannot be dodged by choosing
a good `distmax`. What is reliable is that a *saturated* return — the query radius itself —
proves the true distance is at least that radius. So a zero is re-queried at successively
halved radii, and the first non-zero answer is either the true distance or a proof it exceeds
that probe; both are safe to use. Only zeros pay for the extra queries. The bounding-sphere
gap, being a valid lower bound, floors the result as well.

On one already-generated 45° trajectory this turned "76 of 1606 frames interpenetrating,
closest pair 0.0 mm" into the truth: **no interpenetration at all, closest pair 10.6 mm**.
Real collisions are unaffected — those come back negative, which the guard never touches, and
the 25 mm arm-in-wrist penetration below re-measures identically through it.

### Stepping

Adding the feet to the mode language is one line of representation and three consequences that
are easy to get wrong, so they are worth stating:

1. **A stance placement is shared, exactly like a grasp point.** A foot planted across a run of
   stages owns one `(x, y, yaw)` for the whole run, so it structurally cannot slide; a step is
   the boundary between two such episodes. The first episode of each foot is *pinned* to the
   spawn stance — a reference that starts somewhere the robot is not cannot be played back.
2. **Balance at a transition uses the feet planted on *both* sides of it**, not the union. The
   instant a foot leaves the ground it can no longer be leaned on, so the centre of mass has to
   be over the other foot already. Getting this wrong is not subtle: with the union, the solver
   happily produced double-support keyframes on either end of a swing, and the dynamic pass then
   reported the ZMP 18 cm outside the stance foot. It also means consecutive swings need a
   double-support stage between them — back-to-back swings leave an instant with *no* foot
   common to both sides, i.e. nothing to stand on.
3. **A step needs a reason.** The constraints only say what is *reachable*, and a chair 0.55 m
   away already is, so a purely constraint-driven optimizer takes zero-length steps and the
   stages are dead weight. Each free stance therefore carries a weakly-weighted target: squared
   up to the object, one body-length back from the envelope it sweeps as it turns
   (`ChairSpec.sweep_radius` + `scene.body_reach`, both measured from the models, neither
   chosen). The sequence is then solved *standing still* first and stepped onto that target,
   rather than asking a soft preference to walk a converged standing solution out of its own
   minimum — which it cannot, because a placement shared by seven frames barely moves.

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
   Those durations respect joint and base velocity limits *and* the acceleration each stage's
   own CoM excursion implies — a weight transfer from one foot to the other moves the centre
   of mass 24 cm with every joint below 3 % of its velocity limit, so timing it by velocity
   alone gives it 0.7 s and throws the ZMP half a metre out.
2. **Kinodynamic checks** (`dynamics.py`): the wrench the hands must apply to move the chair as
   commanded, split over the active contacts and tested against friction cone, torsional
   friction and centre-of-pressure bounds; the ZMP of the robot (carrying whatever the hands
   hold, taking the reaction of whatever they push) against *whichever feet are on the ground
   at that instant*; joint velocities and inverse-dynamics torques against the URDF limits,
   with the ground reaction carried by the stance leg alone during a step.
3. **Retiming as the repair** (FARO's `T̄_s`): every violation that scales with acceleration is
   fixed by stretching stage durations, which the generator does automatically. Violations that
   do *not* scale — a grasp that must resist more moment than the contact can hold, a posture
   whose gravity torque exceeds the motors — stop the loop and reject the plan, and the search
   moves to the next candidate. That is FARO's TO filter in the role it actually plays.

### Plan library

Five families: **push** (both palms on one backrest face, floor-supported — cheapest when
the goal is a modest in-plane change), **slide** (pinch two opposing faces and swivel the
chair on the floor — the legs keep carrying the weight, so nothing has to be balanced),
**lift** (pinch and carry, per patch pair, ranked by how laterally the robot can squeeze and
how far it must reach), **rail_lift** (both palms over the backrest top rail), and **liftx2**
— a regrasp: set the chair down, let go, take hold of the face that is *now* lateral,
continue.

Each family is then offered in three stance variants, and `--plan auto` (the default) tries
them in this order:

1. **walk-in** — two steps before the hands do anything, so the grasp is made from a stance
   the robot chose rather than from wherever it spawned. This is the default because it is
   strictly better: reaching a chair half a metre away from the spawn stance spends the
   entire balance margin on the reach itself.
2. **walk-in + pivot** (large displacements only) — a step pair *mid-motion*, chair held
   still, so the stance follows the object through the turn. Unlike a regrasp it keeps the
   contacts, so the chair is never put down.
3. **standing** — the feet-planted plans. Kept as a fallback because they are cheaper to
   filter and perfectly valid for a small displacement, not because they are preferred.
4. **regrasp** (`liftx2`), last: the only option that buys reach neither reaching nor
   stepping can, at the price of releasing the object mid-way and clearing another full set
   of filters.

A step is one single-support stage: the swinging foot's placement variable jumps to the next
episode's, and the interior knots of that stage — which is where the swing actually happens,
since KSO only ever sees the double-support instants at either end — are generated by the
path pass.

## Results

Chair: the staged `data/objects/chair` asset (SANDSBERG, seat 0.48 × 0.42 m at 0.41 m, 0.79 m
tall), whose bounding box sizes the box proxy and whose mesh is drawn on top; 0.2 kg, matching
`run_sim_loop.CHAIR_MASS`. Without a staged asset the hardcoded SANDSBERG-like proxy
(0.40 × 0.40 m at 0.45 m, 0.86 m tall) is used instead. All runs on one CPU, `.venv_sim`.

| Task | Plan chosen | Wall time | Duration | Rejected first | ZMP margin | Contact drift | Twist | Step height |
|---|---|---|---|---|---|---|---|---|
| rotate 45° in place | `walk+rail_lift` | 428 s | 21.2 s | 6 | −0.3 cm | 3.8 mm | 0.6° | 31 mm |
| rotate 90° in place | *none — torque 1.09×* | — | — | all | — | — | — | — |
| rotate 90°, `--no-torque` | `walk+rail_lift` | 240 s | 29.7 s | 6 | −0.7 cm | 4.5 mm | 1.6° | 32 mm |

"Contact drift" is the largest distance a palm moves off its patch surface anywhere in the
sampled trajectory, and "twist" the largest rotation of a palm *on* the object within one
grasp (the keyframes and knots are exact; both are what interpolation between them costs).
"Step height" is how far a swinging foot actually clears the floor.

Independently re-measured on both results, against the same quantities before the three
fixes below:

| | before | 45° after | 90° after |
|---|---|---|---|
| palm twist within a grasp | 108.4° | **0.6°** | **1.6°** |
| closest arm-vs-arm approach | −24.8 mm (222 frames interpenetrating) | **+42.5 mm** | **+21.2 mm** |
| frames interpenetrating, any checked pair | 222 / 1955 | **0 / 1061** | **0 / 1487** |

The `--no-torque` row is the only one that would not be produced by default: it is the 90°
task with the actuation check demoted to advisory (see the limitations). Everything else
about it is checked normally, which is why it is worth measuring — it shows the twist and
self-collision fixes hold up on the harder task, independently of the torque question.

Wall time is single-CPU and is dominated by the plans that get *rejected* — 45° pays for six
failed candidates before `walk+rail_lift` succeeds. It is essentially all finite-difference
Jacobian evaluation. A stepping plan costs roughly twice a standing one to filter: half again
as many stages, plus the stance-continuation solve.

Two things are worth reading off that table.

**A grasp that cannot twist is a materially harder problem.** Fixing the palm's bearing costs
a degree of freedom the optimizer had been quietly spending, and the posture it has to adopt
instead is more expensive: the 45° task went from six stages standing still to an eleven-stage
walk-and-lift, and 90° now exceeds the torque estimate. That is the correct direction — the
cheap solutions it used to find were not physically realisable — but it is not free.

**The joint torque check is now the binding constraint at every distance.** It refuses 90° at
1.09× and the 0.85 m chair at 1.3–1.7×, and 45° passes at 0.89×, i.e. only just. Everything
else — reach, balance, ZMP, friction, grasp moment — has margin. Given how crude that estimate
is (see the limitations), it is the number most worth improving next, and `--no-torque` exists
to look past it in the meantime.

## Outputs

Written to `data/generated_trajectories/<name>/`:

- **`trajectory.npz`** — `t`, full-scene MuJoCo `qpos` (robot + chair, directly replayable),
  `whole_q` (51-DOF Pinocchio layout, the same thing the recorder stores as
  `observation.state`), `dof29`, `hand_dof`, the chair pose in both the canonical proxy frame
  and the staged asset's own frame (the latter is what the sim's `object` free joint takes),
  the per-frame hand `contact` flags, the per-frame `foot_contact` flags and `foot_placement`
  (where each foot is planted, or heading to mid-swing), and stage indices.
- **`motion_lib.pkl`** — SONIC training format (`root_trans_offset`, `root_rot` (xyzw), `dof`
  in MuJoCo joint order, `pose_aa`, `fps`), plus `object_trans` / `object_rot` / `contact` /
  `foot_contact` / `hand_dof` as extra keys, so an object-aware tracking reward — and a foot
  contact reward — has everything in one file.
- **`report.json`** — every candidate plan, which filter rejected it and why, and all
  feasibility margins of the accepted trajectory.
- **`scene.xml`** — the exact MuJoCo scene it was generated against, loadable standalone.

`--replay` plays the result in the MuJoCo viewer as it is generated;
`--replay-from data/generated_trajectories/<name>` plays one written earlier and generates
nothing (space = pause, arrows = step, the current stage is printed as it changes). Both play
the exported `qpos` against the run's own `scene.xml`, so a replay shows exactly what was
written and nothing is re-derived.

`--no-torque` keeps measuring the joint torque ratio but stops it vetoing a plan. The
actuation check is the crudest of the set, and it is currently what refuses a chair at
0.85 m — this is how to look at the trajectory it refuses and judge for yourself.

## Known limitations

- **Steps are quasi-static.** A single-support phase is held to a CoM inside the stance
  sole, so the robot shifts its weight fully over one foot before the other leaves the
  ground and cannot "fall forward" into a step the way a real walk does. That is the right
  conservatism for a reference (it is what a tracking controller can follow from rest), but
  it costs time: each step stretches until the ZMP fits, and the 6 cm-wide sole is what sets
  that. Dynamic stepping would need the momentum terms the centroidal check drops.
- **Where to step is chosen, when to step is not.** The number of steps and which foot moves
  come from the plan library; only the placements are optimized. A search over step counts
  is the natural next thing to hand to the tree search.
- **Balance is checked as a CoM/ZMP condition on the *robot alone***. A robot can legitimately
  lean on what it pushes, but only if that object carries the load — and this chair weighs
  0.2 kg, so the conservative choice is taken. Raising `--chair-mass` does not currently unlock
  leaning; the polygon would have to be extended through the hand contacts.
- **Grasps are modelled as a wrapped grip** (`dynamics.GRIP_FORCE`, `GRASP_LEVER`) rather than
  as a finger-level force closure. The finger trajectory itself is a scripted open/closed ramp,
  not optimized — this is exactly the contact detail the RL controller is expected to supply.
- **Angular-momentum rate is neglected** in the ZMP check, and the torque check splits the
  ground reaction evenly between whichever feet are down. Both are magnitude checks, not
  certificates.
- **The stance target is a heuristic**, even though both of its terms are measured. It says
  "square up to the object, one body-length clear of the envelope it sweeps", which is a
  reasonable thing for a robot to want and demonstrably produces sensible steps — but it is
  a preference the optimizer is nudged toward, not something derived from the task.
- **The chair proxy is a box abstraction** (seat plate, backrest, four legs) sized from the
  staged asset's bounding box. Contact patches are defined on the proxy, so a chair shaped very
  differently from the SANDSBERG proportions needs its `--seat-height` (or the `ChairSpec`
  defaults) adjusted.
- **9 knots per stage is a sweet spot, not a floor.** More knots make the coupled stage solve
  harder to converge within its iteration budget and the path gets *worse*; `--knots` is
  exposed but 12 already degrades.
- **A chair at 0.85 m is refused outright, on leg torque.** `--start 0.85 0 0 --rotate-deg 45`
  clears the kinematics — the walk-in solves, the path comes back with every stage inside
  tolerance — and is then turned down by the actuation check, every candidate, over 26 minutes
  and all 14 plans. The numbers are worth recording because they point the opposite way to the
  obvious guess: the *stepping* plans ask 1.31–1.33× the usable knee/hip torque, the *standing*
  `rail_lift` asks **1.72×**. Walking to the chair is the cheaper option even though single
  support puts the whole ground reaction through one leg — leaning out over a chair that far
  away costs more than stepping to it does. What the robot cannot currently do is either.

  Whether that verdict is the robot or the model is genuinely open. `TORQUE_FRACTION` allows
  only 80 % of the URDF limit, the check neglects angular-momentum rate, and it applies the
  ground reaction at the sole centre — it is a magnitude estimate, not a certificate. It is
  now the binding constraint on how far the robot will walk, and it is the first thing to
  re-examine before concluding the G1 cannot reach a chair at 0.85 m.

  `--no-torque` exists for exactly that examination, and what it produces argues the estimate
  is the suspect party: with the veto lifted the *first* stepping candidate is accepted, and
  it is the cleanest trajectory the generator has made — 20 cm steps, the pelvis travelling
  0.00 → 0.17 m, 3 cm of foot clearance, 0.4 mm of contact drift and the ZMP on the polygon
  edge. Nothing about it looks like a trajectory that is 34 % beyond the machine; it looks
  like an ordinary walk being scored by a check that puts the entire body weight through one
  sole at a point. Deciding this properly needs the momentum terms the check drops.

## Next steps, in order of expected value

1. Train the tracking controller on a generated reference and measure what actually survives —
   the whole point is that the RL policy absorbs contact detail, and that assumption is
   currently untested.
2. Replace the plan library with FARO's actual tree search over mode sequences (cost-based UCT
   with progressive widening). Stepping is what makes that worth doing: the number of steps,
   which foot leads and where each lands is a genuinely large space, and it is currently
   enumerated by hand. The filter cascade and cache the search needs are already here.
3. Analytic derivatives (pinocchio, or casadi if it is ever added to the sim env). The
   finite-difference Jacobian is the entire runtime cost; everything else is milliseconds —
   and stepping roughly doubled the number of variables, so this matters more than it did.
