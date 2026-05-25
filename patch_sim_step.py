import re

with open("gear_sonic/utils/mujoco_sim/base_sim.py", "r") as f:
    code = f.read()

old_str = """        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[:self.num_body_dof] = body_torques
        if self.num_hand_dof > 0:
            # Only assign to actuated hand joints (first num_hand_dof elements)
            # Remaining elements are mimic joints controlled by the actuated ones
            self.torques[self.num_body_dof : self.num_body_dof + self.num_hand_dof] = hand_torques[: self.num_hand_dof]
            self.torques[self.num_body_dof + self.num_hand_dof :] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        # breakpoint()
        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)"""

new_str = """        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        
        logic_torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        logic_torques[:self.num_body_dof] = body_torques
        if self.num_hand_dof > 0:
            logic_torques[self.num_body_dof : self.num_body_dof + self.num_hand_dof] = hand_torques[: self.num_hand_dof]
            logic_torques[self.num_body_dof + self.num_hand_dof :] = hand_torques[self.num_hand_dof :]

        logic_torques = np.clip(logic_torques, -self.torque_limit, self.torque_limit)
        self.torques = logic_torques

        ctrl = np.zeros(self.mj_model.nu)
        
        for i, idx in enumerate(self.body_actuator_index):
            if idx != -1 and i < len(body_torques):
                ctrl[idx] = logic_torques[i]
                
        if self.num_hand_dof > 0:
            for i, idx in enumerate(self.left_hand_actuator_index):
                if idx != -1 and i < self.num_hand_dof:
                    ctrl[idx] = logic_torques[self.num_body_dof + i]
            for i, idx in enumerate(self.right_hand_actuator_index):
                if idx != -1 and i < self.num_hand_dof:
                    ctrl[idx] = logic_torques[self.num_body_dof + self.num_hand_dof + i]

        if self.config["FREE_BASE"]:
            if len(ctrl) + 6 == len(self.mj_data.ctrl):
                self.mj_data.ctrl = np.concatenate((np.zeros(6), ctrl))
            else:
                self.mj_data.ctrl = ctrl
        else:
            self.mj_data.ctrl = ctrl
        mujoco.mj_step(self.mj_model, self.mj_data)"""

if old_str in code:
    with open("gear_sonic/utils/mujoco_sim/base_sim.py", "w") as f:
        f.write(code.replace(old_str, new_str))
    print("Patched successfully")
else:
    print("Could not find the string to replace.")

