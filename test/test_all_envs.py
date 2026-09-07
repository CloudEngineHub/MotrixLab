# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry


def test_all_demos():
    print("Start testing:")

    num_envs = [1, 2]

    all_envs = list(registry.list_registered_envs())

    total_count = 0
    failed_count = 0
    for num_env in num_envs:
        for env_name in all_envs:
            total_count += 1

            try:
                # Create environment (manager envs inject the default backend)
                env = registry.make(env_name, num_envs=num_env)

                action_space = env.action_space
                action = np.zeros((env.num_envs, *action_space.shape), dtype=action_space.dtype)

                for _ in range(10):
                    env.step(action)
                print(f"{env_name} pass.")

            except Exception as e:
                failed_count += 1
                print(f"{env_name} fail.")
                print(e)

    print(f"\nComplete {total_count} tests.\n{total_count - failed_count} cases pass.")
    assert failed_count == 0, f"{failed_count} cases failed"
