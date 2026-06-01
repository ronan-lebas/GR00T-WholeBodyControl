# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.10.9",
#     "numpy>=2.4.5",
#     "pandas>=3.0.3",
#     "pyarrow>=24.0.0",
# ]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# auto-select latest timestamped recording folder under `outputs/`
base_outputs = Path('outputs')
if not base_outputs.exists():
	raise FileNotFoundError('outputs/ directory not found')
subdirs = [p for p in base_outputs.iterdir() if p.is_dir()]
if not subdirs:
	raise FileNotFoundError('no recording folders found in outputs/')
# timestamp folders sort lexicographically when using YYYY-MM-DD-HH-MM-SS
latest = sorted(subdirs)[-1]
# find parquet files under latest/data (recursive)
parquets = list(latest.rglob('*.parquet'))
if not parquets:
	raise FileNotFoundError(f'no parquet files under {latest}')
# prefer episode_000000.parquet if present
chosen = None
for p in parquets:
	if p.name == 'episode_000000.parquet':
		chosen = p
		break
if chosen is None:
	chosen = sorted(parquets)[-1]
print(f'Using recording: {chosen}')
df = pd.read_parquet(chosen)

# Stack commonly used arrays
obs_joint = np.stack(df['observation.state'].values)
act_joint = np.stack(df['action.wbc'].values)
eef_state = np.stack(df['observation.eef_state'].values)
proj_grav = np.stack(df['observation.projected_gravity'].values)

# Teleop hand joints (optional in dataset): the target sent by the streamer
left_hand = None
right_hand = None
if 'teleop.left_hand_joints' in df.columns:
	left_hand = np.stack(df['teleop.left_hand_joints'].values)
if 'teleop.right_hand_joints' in df.columns:
	right_hand = np.stack(df['teleop.right_hand_joints'].values)

# Infer hand DOF count from the data: 6 = BrainCo, 7 = DEX3.
if left_hand is not None:
	n_hand_dof = left_hand.shape[1]
elif right_hand is not None:
	n_hand_dof = right_hand.shape[1]
else:
	# action.wbc total size minus 29 body joints, split evenly
	n_hand_dof = (act_joint.shape[1] - 29) // 2
hand_type = "brainco" if n_hand_dof == 6 else "dex3"
print(f'Hand DOF: {n_hand_dof} ({hand_type.upper()})')

# Hardcoded actuated joint indices in the full 43/51-DOF Pinocchio q vector.
# Hand joints follow the arm joints (not all body joints), so the indices are NOT
# the naive [29..34] / [35..41]. Verified from instantiate_g1_robot_model().
if hand_type == "brainco":
	# 51-DOF model: legs(12) + waist(3) + L-arm(7) + L-hand(11) + R-arm(7) + R-hand(11)
	# L-hand motor joints (order: thumb_meta, thumb_prox, index_prox, middle_prox, ring_prox, pinky_prox)
	left_motor_map  = [30, 31, 22, 24, 28, 26]
	right_motor_map = [48, 49, 40, 42, 46, 44]
	motor_names = ['thumb_meta', 'thumb_prox', 'index_prox', 'middle_prox', 'ring_prox', 'pinky_prox']
else:  # dex3
	# 43-DOF model: legs(12) + waist(3) + L-arm(7) + L-hand(7) + R-arm(7) + R-hand(7)
	# L-hand motor joints (order: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1)
	left_motor_map  = [26, 27, 28, 22, 23, 24, 25]
	right_motor_map = [40, 41, 42, 36, 37, 38, 39]
	motor_names = ['thumb_0', 'thumb_1', 'thumb_2', 'index_0', 'index_1', 'middle_0', 'middle_1']

# Each subplot shows three signals per motor joint:
#   teleop target  (blue solid)  — what you commanded from the streamer
#   action.wbc     (orange dash) — what the WBC controller output (same index in q vector)
#   obs.state      (green dot)   — actual sensor reading from the robot (same index)
n_pairs = 2 * n_hand_dof
n_plots = n_pairs + 1  # +1 for EEF
cols = 3
rows = int(np.ceil(n_plots / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=True)
axes = axes.flatten()

# plot left hand motors
for i in range(n_hand_dof):
	ax = axes[i]
	q_idx = left_motor_map[i]
	if left_hand is not None:
		ax.plot(left_hand[:, i], label='teleop target', color='C0')
	else:
		ax.text(0.5, 0.5, 'no teleop.left_hand_joints', ha='center')
	ax.plot(act_joint[:, q_idx], label='action.wbc (cmd)', linestyle='--', color='C1', alpha=0.9)
	ax.plot(obs_joint[:, q_idx], label='obs.state (actual)', linestyle=':', color='C2', alpha=0.9)
	ax.set_title(f'L {motor_names[i]}  q[{q_idx}]')
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize='small')

# plot right hand motors
for j in range(n_hand_dof):
	ax = axes[n_hand_dof + j]
	q_idx = right_motor_map[j]
	if right_hand is not None:
		ax.plot(right_hand[:, j], label='teleop target', color='C0')
	else:
		ax.text(0.5, 0.5, 'no teleop.right_hand_joints', ha='center')
	ax.plot(act_joint[:, q_idx], label='action.wbc (cmd)', linestyle='--', color='C1', alpha=0.9)
	ax.plot(obs_joint[:, q_idx], label='obs.state (actual)', linestyle=':', color='C2', alpha=0.9)
	ax.set_title(f'R {motor_names[j]}  q[{q_idx}]')
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize='small')

# final subplot: EEF
ax_eef = axes[n_pairs]
ax_eef.plot(eef_state[:, 0], label='X')
ax_eef.plot(eef_state[:, 1], label='Y')
ax_eef.plot(eef_state[:, 2], label='Z')
ax_eef.set_title('End Effector Position')
ax_eef.set_xlabel('Time Step (Frames)')
ax_eef.set_ylabel('Position')
ax_eef.legend()
ax_eef.grid(True, alpha=0.3)

# hide any extra axes
for k in range(n_plots, len(axes)):
	fig.delaxes(axes[k])

plt.tight_layout()
plt.show()
