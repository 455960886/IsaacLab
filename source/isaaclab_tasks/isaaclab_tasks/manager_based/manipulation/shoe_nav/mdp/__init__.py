# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms specific to the shoe-navigation environment.

Re-exports the global IsaacLab MDP terms (so things like ``mdp.time_out``,
``mdp.reset_root_state_uniform``, ``mdp.JointVelocityActionCfg``,
``mdp.pointnet_features`` etc. resolve here as well), and adds task-specific
terms in the local sub-modules.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .events import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
