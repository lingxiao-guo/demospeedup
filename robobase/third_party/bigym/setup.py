from setuptools import setup, find_packages

setup(
    name="bigym",
    version="0.0.0",
    packages=find_packages(), 
    include_package_data=True,
    install_requires=[
    # includes bugfix in mujoco_rendering
    "gymnasium @ git+ssh://git@github.com/stepjam/Gymnasium.git@0.29.2",
    # pyquaternion doesn't support 2.x yet
    "numpy==1.26.*",
    "safetensors==0.3.3",
    # WARNING: recorded demos might break when updating Mujoco
    "mujoco==3.1.5",
    # needed for pyMJCF
    "dm_control==1.0.19",
    "imageio",
    "pyquaternion",
    "mujoco_utils",
    "wget",
    "mojo @ git+ssh://git@github.com/stepjam/mojo.git@0.1.1",
    "pyyaml",
    "dearpygui",
    "pyopenxr",
    "moviepy",
    "pygame",
    "opencv-python",
    "matplotlib",
    ],
)
