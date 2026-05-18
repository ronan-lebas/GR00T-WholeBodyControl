import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet('outputs/2026-05-18-17-09-24/data/chunk-000/episode_000000.parquet')

obs_joint = np.stack(df['observation.state'].values)
act_joint = np.stack(df['action.wbc'].values)
eef_state = np.stack(df['observation.eef_state'].values)
proj_grav = np.stack(df['observation.projected_gravity'].values)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

axes[0, 0].plot(obs_joint[:, 0], label='Observed', color='blue')
axes[0, 0].plot(act_joint[:, 0], label='Target', color='red', linestyle='--')
axes[0, 0].set_title('Joint 0 Tracking')
axes[0, 0].set_ylabel('Position (rad)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(obs_joint[:, 1], label='Observed', color='blue')
axes[0, 1].plot(act_joint[:, 1], label='Target', color='red', linestyle='--')
axes[0, 1].set_title('Joint 1 Tracking')
axes[0, 1].set_ylabel('Position (rad)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(eef_state[:, 0], label='X')
axes[1, 0].plot(eef_state[:, 1], label='Y')
axes[1, 0].plot(eef_state[:, 2], label='Z')
axes[1, 0].set_title('End Effector Position')
axes[1, 0].set_xlabel('Time Step (Frames)')
axes[1, 0].set_ylabel('Position')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(proj_grav[:, 0], label='X')
axes[1, 1].plot(proj_grav[:, 1], label='Y')
axes[1, 1].plot(proj_grav[:, 2], label='Z')
axes[1, 1].set_title('Projected Gravity (Base Tilt)')
axes[1, 1].set_xlabel('Time Step (Frames)')
axes[1, 1].set_ylabel('Vector Component')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()