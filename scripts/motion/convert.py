# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert motion npz files from public source formats to MotrixLab v1.

The private holosoma / xMimic converters live under ``scripts/private/`` (internal,
not published); run ``scripts/private/convert.py`` for those.

Examples:
    # LAFAN1_Retargeting_Dataset G1 csv -> MotrixLab v1 (forward-kinematics bake)
    uv run scripts/motion/convert.py \\
        --from lafan \\
        --input /path/to/LAFAN1_Retargeting_Dataset/g1/dance1_subject1.csv \\
        --output motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject1.npz \\
        --output-fps 50
"""

from absl import app, flags

from motrix_envs.motion.converters import CONVERTERS

_FLAGS_FROM = flags.DEFINE_string(
    "from", None, f"Source format (one of: {sorted(CONVERTERS)}).", required=True, short_name="f"
)
_FLAGS_INPUT = flags.DEFINE_string("input", None, "Path to source .npz/.csv file.", required=True, short_name="i")
_FLAGS_OUTPUT = flags.DEFINE_string("output", None, "Path to destination .npz file.", required=True, short_name="o")
# LAFAN-only options (ignored by converters that don't accept them).
_FLAGS_ROBOT = flags.DEFINE_string("robot", "g1", "[lafan] Robot type whose model drives forward kinematics.")
_FLAGS_INPUT_FPS = flags.DEFINE_float("input-fps", 30.0, "[lafan] Source frame rate of the csv.")
_FLAGS_OUTPUT_FPS = flags.DEFINE_float("output-fps", 50.0, "[lafan] Target frame rate of the output npz.")
_FLAGS_START_SEC = flags.DEFINE_float("start-sec", 0.0, "[lafan] Trim: start time in seconds (default 0).")
_FLAGS_END_SEC = flags.DEFINE_float("end-sec", None, "[lafan] Trim: end time in seconds (default: end of clip).")
_FLAGS_MODEL_FILE = flags.DEFINE_string("model-file", None, "[lafan] Override MotrixSim model xml path.")

# Per-source keyword options forwarded to the converter beyond (input, output).
_SOURCE_OPTS = {
    "lafan": lambda: {
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
