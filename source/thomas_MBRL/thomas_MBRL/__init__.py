# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

from gymnasium.envs.registration import register

register(
    id="Thomas-Go2-Velocity-v0",
    entry_point="thomas_MBRL.envs.go2_velocity_env:MyGo2VelocityEnv",
)