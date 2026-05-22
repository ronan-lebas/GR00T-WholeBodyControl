import os
import mujoco

# 1. Define paths
urdf_path = "gear_sonic/data/robot_model/model_data/g1/with_brainco/g1_29dof_with_hand.urdf"
urdf_dir = os.path.dirname(urdf_path)
urdf_file = os.path.basename(urdf_path)

# 2. Change working directory to the URDF folder 
# (This ensures MuJoCo resolves local files properly)
os.chdir(urdf_dir)

# 3. Read the URDF file as text
with open(urdf_file, "r") as f:
    xml_string = f.read()

# 4. Fix the double path issue on the fly
# We strip the "meshes/" prefix from the filenames so MuJoCo's compiler doesn't double it
xml_string = xml_string.replace('filename="meshes/', 'filename="')
xml_string = xml_string.replace("filename='meshes/", "filename='")

# 5. Load the model from the corrected string
model = mujoco.MjModel.from_xml_string(xml_string)

# 6. Save the new MJCF file (this will save in the 'with_brainco' folder)
output_filename = "g1_29dof_with_hand.xml"
mujoco.mj_saveLastXML(output_filename, model)

print(f"Success! Saved to {os.path.abspath(output_filename)}")