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

# Teleop hand joints (optional in dataset)
left_hand = None
right_hand = None
if 'teleop.left_hand_joints' in df.columns:
	left_hand = np.stack(df['teleop.left_hand_joints'].values)
if 'teleop.right_hand_joints' in df.columns:
	right_hand = np.stack(df['teleop.right_hand_joints'].values)

# Infer hand DOF count from the data: 6 = BrainCo, 7 = DEX3.
# Fall back to half of the hand columns in action.wbc if no teleop data.
if left_hand is not None:
	n_hand_dof = left_hand.shape[1]
elif right_hand is not None:
	n_hand_dof = right_hand.shape[1]
else:
	# action.wbc has 29 body joints + left hand + right hand
	n_hand_dof = (act_joint.shape[1] - 29) // 2
print(f'Hand DOF: {n_hand_dof} ({"BrainCo" if n_hand_dof == 6 else "DEX3"})')

# action.wbc layout: [0..28] body joints, [29..29+n) left hand, [29+n..29+2n) right hand
left_map  = list(range(29, 29 + n_hand_dof))
right_map = list(range(29 + n_hand_dof, 29 + 2 * n_hand_dof))

# Create separate subplot for each teleop joint vs its corresponding action.wbc column
n_pairs = 2 * n_hand_dof
n_plots = n_pairs + 1  # +1 for EEF plotting
cols = 3
rows = int(np.ceil(n_plots / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=True)
axes = axes.flatten()

# plot left hand pairs
for i in range(n_hand_dof):
	ax = axes[i]
	act_idx = left_map[i]
	if left_hand is not None:
		ax.plot(left_hand[:, i], label=f'left_teleop_{i}', color='C0')
	else:
		ax.text(0.5, 0.5, 'no teleop.left_hand_joints', ha='center')
	ax.plot(act_joint[:, act_idx], label=f'action.wbc[{act_idx}]', linestyle='--', color='C1', alpha=0.9)
	ax.set_title(f'Left joint {i}  — action idx {act_idx}')
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize='small')

# plot right hand pairs
for j in range(n_hand_dof):
	ax = axes[n_hand_dof + j]
	act_idx = right_map[j]
	if right_hand is not None:
		ax.plot(right_hand[:, j], label=f'right_teleop_{j}', color='C0')
	else:
		ax.text(0.5, 0.5, 'no teleop.right_hand_joints', ha='center')
	ax.plot(act_joint[:, act_idx], label=f'action.wbc[{act_idx}]', linestyle='--', color='C1', alpha=0.9)
	ax.set_title(f'Right joint {j}  — action idx {act_idx}')
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