# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert motion npz files from PRIVATE source formats to MotrixLab v1.

Internal tooling for source formats that are not published with the package
(Holosoma / xMimic / delivered K1 MuJoCo NPZ). The public LAFAN1 converter is at
``scripts/motion/convert.py``.

Examples:
    # Holosoma npz -> MotrixLab v1 (pure reformat)
    uv run scripts/private/convert.py \\
        --from holosoma \\
        --input /path/to/holosoma.npz \\
        --output motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/converted.npz

    # xMimic Dex-EVT npz -> MotrixLab v1 (forward-kinematics bake)
    uv run scripts/private/convert.py \\
        --from xmimic \\
        --input /path/to/xmimic_dance.npz \\
        --output motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/dex_evt/dance1.npz \\
        --output-fps 50
"""

from absl import app, flags

# These converter modules live alongside this script (scripts/private/), which is
# on sys.path when the script is run directly.
from holosoma_converter import convert_holosoma
from k1_mujoco_converter import convert_k1_mujoco
from xmimic_converter import convert_xmimic

CONVERTERS = {
    "holosoma": convert_holosoma,
    "k1-mujoco": convert_k1_mujoco,
    "xmimic": convert_xmimic,
}

_FLAGS_FROM = flags.DEFINE_string(
    "from", None, f"Source format (one of: {sorted(CONVERTERS)}).", required=True, short_name="f"
)
_FLAGS_INPUT = flags.DEFINE_string("input", None, "Path to source .npz file.", required=True, short_name="i")
_FLAGS_OUTPUT = flags.DEFINE_string("output", None, "Path to destination .npz file.", required=True, short_name="o")
# FK-bake options (ignored by converters that don't accept them).
_FLAGS_ROBOT = flags.DEFINE_string("robot", "dex_evt", "[xmimic] Robot type whose model drives forward kinematics.")
_FLAGS_INPUT_FPS = flags.DEFINE_float("input-fps", None, "Source frame rate override (default: stored in npz).")
_FLAGS_OUTPUT_FPS = flags.DEFINE_float("output-fps", 50.0, "Target frame rate of the output npz.")
_FLAGS_START_SEC = flags.DEFINE_float("start-sec", 0.0, "Trim start time in seconds (default 0).")
_FLAGS_END_SEC = flags.DEFINE_float("end-sec", None, "Trim end time in seconds (default: clip end).")
_FLAGS_MODEL_FILE = flags.DEFINE_string("model-file", None, "Override the MotrixSim model xml path.")

# Per-source keyword options forwarded to the converter beyond (input, output).
_SOURCE_OPTS = {
    "k1-mujoco": lambda: {
        "input_fps": _FLAGS_INPUT_FPS.value,
        "output_fps": _FLAGS_OUTPUT_FPS.value,
        "start_sec": _FLAGS_START_SEC.value,
        "end_sec": _FLAGS_END_SEC.value,
        "model_file": _FLAGS_MODEL_FILE.value,
    },
    "xmimic": lambda: {
        "robot": _FLAGS_ROBOT.value,
        "input_fps": _FLAGS_INPUT_FPS.value,
        "output_fps": _FLAGS_OUTPUT_FPS.value,
        "start_sec": _FLAGS_START_SEC.value,
        "end_sec": _FLAGS_END_SEC.value,
        "model_file": _FLAGS_MODEL_FILE.value,
    },
}


def main(argv):
    del argv  # unused

    source = _FLAGS_FROM.value
    if source not in CONVERTERS:
        raise SystemExit(f"Unknown source format {source!r}; available: {sorted(CONVERTERS)}")

    opts = _SOURCE_OPTS.get(source, lambda: {})()
    stats = CONVERTERS[source](_FLAGS_INPUT.value, _FLAGS_OUTPUT.value, **opts)
    print(
        f"Converted {_FLAGS_INPUT.value} -> {stats['output_path']}: "
        f"{stats['num_frames']} frames, {stats['num_joints']} joints, "
        f"{stats['num_bodies']} bodies, has_object={stats['has_object']}"
    )


if __name__ == "__main__":
    app.run(main)
