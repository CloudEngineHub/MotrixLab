# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _pin_absl_log_stream_to_real_stderr():
    """Point absl's root log handler at the real stderr after each test.

    Importing tensorboard pulls in absl.logging, whose root handler keeps the
    ``sys.stderr`` it saw at import time — pytest's global capture file. Any
    later hydra app reconfigures logging via dictConfig, which closes old
    handlers; absl's handler close would then close pytest's capture file and
    break output capture for every remaining test in the session
    (``ValueError: I/O operation on closed file``).
    """
    yield
    absl_logging = sys.modules.get("absl.logging")
    if absl_logging is not None:
        absl_logging.get_absl_handler().python_handler.stream = sys.__stderr__


@pytest.fixture(autouse=True)
def _drop_cv2_dir_from_sys_path():
    """Remove cv2's directory from sys.path after each test.

    Importing tensorboard pulls in keras, which imports cv2. When that import
    fails (CI images without libGL), opencv's bootstrap dies after inserting
    its own directory into sys.path and before restoring it. Spawned
    multiprocessing children then resolve stdlib ``typing`` to
    ``cv2/typing`` and crash at startup with an ImportError.
    """
    yield
    sys.path[:] = [entry for entry in sys.path if os.path.basename(os.path.normpath(entry)) != "cv2"]
