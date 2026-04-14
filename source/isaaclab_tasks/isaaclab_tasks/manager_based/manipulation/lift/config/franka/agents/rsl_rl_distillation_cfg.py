# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL runner config for student-teacher distillation.

Network sizes must match the trained checkpoints:
  student_hidden_dims  — matches the old PPO actor [512, 256, 128, 64]
  teacher_hidden_dims  — matches the teacher PPO actor [256, 128, 64]

The teacher weights are loaded from a teacher PPO checkpoint before training
starts (via --load or runner.load()). The student weights are randomly
initialised and trained by behaviour cloning against the frozen teacher.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
)


@configclass
class LiftCubeDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 10
    experiment_name = "coarse_arm_lift_student"
    empirical_normalization = True

    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacher",
        init_noise_std=0.1,
        # Must match the PPO actor architecture used during student (image) training
        student_hidden_dims=[512, 256, 128, 64],
        # Must match exactly the teacher PPO actor architecture in rsl_rl_ppo_cfg_teacher.py
        teacher_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=1,
        learning_rate=1e-3,
        # gradient_length = num_steps_per_env: accumulate all 24 timesteps into
        # one gradient update per epoch (correct for a non-recurrent student).
        gradient_length=24,
    )
