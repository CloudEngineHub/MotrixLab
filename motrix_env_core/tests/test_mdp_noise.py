# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from motrix_env_core.mdp.noise import add_uniform_noise


def test_uniform_noise_consumes_one_sequence_across_consecutive_calls() -> None:
    combined_state = np.asarray([123], dtype=np.uint64)
    combined = np.zeros((6,), dtype=np.float32)
    add_uniform_noise(combined, 1.0, combined_state)

    split_state = np.asarray([123], dtype=np.uint64)
    first = np.zeros((2,), dtype=np.float32)
    second = np.zeros((4,), dtype=np.float32)
    add_uniform_noise(first, 1.0, split_state)
    add_uniform_noise(second, 1.0, split_state)

    np.testing.assert_array_equal(np.concatenate((first, second)), combined)
    np.testing.assert_array_equal(split_state, combined_state)
