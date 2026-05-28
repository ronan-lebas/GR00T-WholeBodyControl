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

df = pd.read_parquet('outputs/2026-05-28-12-18-42/data/chunk-000/episode_000000.parquet')

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

def _deterministic_hand_map(start_idx: int, teleop_hand: np.ndarray | None):
	"""Return deterministic mapping for teleop hand joints into action.wbc columns.

	left hand starts at index 29, right hand at 29+7.
	"""
	if teleop_hand is None:
		return None
	n = teleop_hand.shape[1]
	return list(range(start_idx, start_idx + n))


# deterministic mapping: left starts at 29, right at 29+7
# deterministic action indices for hands (fixed mapping)
left_map = list(range(29, 29 + 7))
right_map = list(range(29 + 7, 29 + 14))

# Create separate subplot for each teleop joint vs its corresponding action.wbc column
n_pairs = 14
n_plots = n_pairs + 1  # +1 for EEF plotting
cols = 3
rows = int(np.ceil(n_plots / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=True)
axes = axes.flatten()

# plot left hand pairs (indices 0..6)
for i in range(7):
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

# plot right hand pairs (indices 7..13)
for j in range(7):
	ax = axes[7 + j]
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