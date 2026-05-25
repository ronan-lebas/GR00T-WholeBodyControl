import re

with open("gear_sonic/utils/mujoco_sim/base_sim.py", "r") as f:
    code = f.read()

old_str = """        # breakpoint()
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]


        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]


            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]


            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]


            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]"""

new_str = """        # breakpoint()
        obs["body_tau_est"] = np.zeros(len(self.body_joint_index))
        for i, act_idx in enumerate(self.body_actuator_index):
            if act_idx != -1:
                obs["body_tau_est"][i] = self.mj_data.actuator_force[act_idx]


        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]


            obs["left_hand_tau_est"] = np.zeros(len(self.left_hand_index))
            for i, act_idx in enumerate(self.left_hand_actuator_index):
                if act_idx != -1:
                    obs["left_hand_tau_est"][i] = self.mj_data.actuator_force[act_idx]


            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]


            obs["right_hand_tau_est"] = np.zeros(len(self.right_hand_index))
            for i, act_idx in enumerate(self.right_hand_actuator_index):
                if act_idx != -1:
                    obs["right_hand_tau_est"][i] = self.mj_data.actuator_force[act_idx]"""

if old_str in code:
    with open("gear_sonic/utils/mujoco_sim/base_sim.py", "w") as f:
        f.write(code.replace(old_str, new_str))
    print("Patched successfully")
else:
    print("Could not find the string to replace.")

