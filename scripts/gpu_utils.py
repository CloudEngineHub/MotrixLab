# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pynvml


def monitor_gpu_utilization(stop_event, gpu_index=0, interval=1.0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    utilization_samples = []

    while not stop_event.is_set():
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        utilization_samples.append(util.gpu)
        stop_event.wait(interval)

    pynvml.nvmlShutdown()

    if utilization_samples:
        data = jnp.array(utilization_samples)
        print(f"GPU utilization statistics over {len(data)} samples:")
        print(f"  Mean: {jnp.mean(data):.2f}%")
        print(f"  Max : {jnp.max(data):.2f}%")
        print(f"  Min : {jnp.min(data):.2f}%")
        print(f"  Median : {jnp.median(data):.2f}%")
    else:
        print("No GPU utilization samples recorded.")
