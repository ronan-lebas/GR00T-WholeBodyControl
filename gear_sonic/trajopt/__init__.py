"""Trajectory optimization for whole-body chair manipulation (FARO-style).

Generates a reference trajectory — robot joints + object pose — for a commanded chair
displacement ("rotate the chair 45 deg about z"), to be tracked by an RL controller.
The reference only has to be *kinematically exact* on the contacts and *quasi-statically
plausible*; contact micro-mechanics are left to the tracking policy.

The structure follows FARO (Ciebielski et al., "FARO: Feasibility-Aware Robot Motion
Optimization", arXiv:2607.18362), adapted to this repo's dependency set (MuJoCo + scipy;
no casadi/acados):

    plans.py    contact-mode sequences (FARO §II-A) and the hardcoded plan library.
                An interface is a palm *or a foot*, so a plan says where the robot
                steps as well as what it grasps
    chair.py    the manipulated object: box proxy + semantic contact patches
    scene.py    MuJoCo scene (G1 + chair) with FK / distance / limit queries
    kso.py      mode + edge feasibility filters (FARO eq. 14) and the kinematic
                sequence optimization over keyframes (eq. 15), whose decision
                variables include the in-patch grasp points and the stance placements
    trajectory.py  stage retiming and the dense per-frame IK polish that turns
                keyframes into a 50 Hz contact-consistent trajectory
    dynamics.py quasi-static / centroidal feasibility pass replacing FARO's full
                trajectory optimization (eq. 17): ZMP, contact wrenches, torques
    export.py   trajectory.npz + motion_lib.pkl + report.json

Entry point: ``gear_sonic/scripts/generate_chair_trajectory.py``.
"""

from gear_sonic.trajopt.chair import ChairSpec, ContactPatch
from gear_sonic.trajopt.plans import ContactMode, ContactPlan, candidate_plans
from gear_sonic.trajopt.scene import TrajOptScene

__all__ = [
    "ChairSpec",
    "ContactPatch",
    "ContactMode",
    "ContactPlan",
    "TrajOptScene",
    "candidate_plans",
]
