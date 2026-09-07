# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task bootstrap and deployment CLI configuration tests."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from hydra import compose, initialize_config_dir
from onnx import TensorProto, helper, numpy_helper

from motrix_deploy.artifact import (
    ControlSpec,
    DeploymentManifest,
    PolicySpec,
    SourceSpec,
    TaskSpec,
    sha256_bytes,
    write_artifact,
)
from motrix_deploy.contracts import RobotSpec, TensorSpec

ROOT = Path(__file__).parents[2]
ROUGH_CONFIG_PATH = ROOT / "configs/deploy/sim2sim/go2_walk_sim2sim.yaml"
FLAT_CONFIG_PATH = ROOT / "configs/deploy/sim2sim/go2_walk_flat_sim2sim.yaml"
REAL_CONFIG_PATH = ROOT / "configs/deploy/sim2real/go2_walk_flat_sim2real.yaml"


def _rollout_result(stdout: str) -> dict[str, object]:
    json_start = stdout.find("{")
    assert json_start >= 0, stdout
    return json.loads(stdout[json_start:])


def test_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from motrix_deploy_tasks import main; raise SystemExit(main())",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "artifact: ???" in result.stdout
    assert "scene: go2-walk-rough" in result.stdout


def test_sim2real_cli_help_selects_hardware_recipe() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from motrix_deploy_tasks import main; raise SystemExit(main())",
            "sim2real",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "name: unitree_go2" in result.stdout
    assert "confirm: false" in result.stdout


def test_import_registers_go2_task_with_motrix_deploy() -> None:
    script = (
        "from motrix_deploy.task import registered_tasks; "
        "assert 'go2_walk/v1' not in registered_tasks(); "
        "import motrix_deploy_tasks; "
        "assert 'go2_walk/v1' in registered_tasks()"
    )

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("config_path", "scene", "base_height"),
    [
        (ROUGH_CONFIG_PATH, "go2-walk-rough", 0.42),
        (FLAT_CONFIG_PATH, "go2-walk-flat", 0.331),
    ],
)
def test_sim2sim_hydra_configs_contain_all_runtime_parameters(
    config_path: Path,
    scene: str,
    base_height: float,
) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        cfg = compose(
            config_name=config_path.stem,
            overrides=["artifact=artifact.deploy"],
        )

    assert cfg.artifact == "artifact.deploy"
    assert cfg.backend.name == "mujoco"
    assert cfg.backend.scene == scene
    assert cfg.backend.sim_dt == 0.002
    assert cfg.backend.solver_iterations == 100
    assert cfg.backend.base_position[2] == pytest.approx(base_height)
    assert cfg.viewer is True


def test_workspace_bootstrap_keeps_default_path_with_explicit_flat_config_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import motrix_deploy_tasks

    captured: list[str] = []

    def capture(args: list[str]) -> int:
        captured.extend(args)
        return 0

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(motrix_deploy_tasks, "deploy_main", capture)

    result: Any = motrix_deploy_tasks.main(
        ["sim2sim", "--config-name", "go2_walk_flat_sim2sim", "artifact=artifact.deploy"]
    )

    assert result == 0
    assert captured.count("--config-name") == 1
    assert captured[captured.index("--config-name") + 1] == "go2_walk_flat_sim2sim"
    assert captured[captured.index("--config-path") + 1] == str(ROOT / "configs/deploy/sim2sim")


def test_sim2real_hydra_config_contains_hardware_guards() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(REAL_CONFIG_PATH.parent)):
        cfg = compose(
            config_name=REAL_CONFIG_PATH.stem,
            overrides=[
                "artifact=artifact.deploy",
                "backend.network_interface=enp3s0",
            ],
        )

    assert cfg.backend.name == "unitree_go2"
    assert cfg.rollout.steps is None
    assert cfg.rollout.duration_s == 300.0
    assert cfg.hardware.confirm is False
    assert cfg.realtime is True
    assert cfg.viewer is False
    assert cfg.command.source == "gamepad"
    assert cfg.backend.kp == 50
    assert cfg.backend.kd == 1
    assert cfg.backend.lie_down_button == "B"
    assert cfg.backend.lie_down_duration_s == pytest.approx(2.0)
    assert cfg.command.gamepad.deadman_button == "L1"
    assert cfg.command.gamepad.invert_linear_x is False
    assert cfg.command.gamepad.invert_linear_y is True
    assert cfg.command.gamepad.invert_yaw is True


def test_sim2real_hydra_config_accepts_explicit_gain_overrides() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(REAL_CONFIG_PATH.parent)):
        cfg = compose(
            config_name=REAL_CONFIG_PATH.stem,
            overrides=[
                "artifact=artifact.deploy",
                "backend.kp=25.0",
                "backend.kd=[0.4,0.4,0.4,0.5,0.5,0.5,0.6,0.6,0.6,0.7,0.7,0.7]",
            ],
        )

    assert cfg.backend.kp == pytest.approx(25.0)
    assert list(cfg.backend.kd) == pytest.approx([0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7])


def test_headless_sim2sim_cli_runs_deterministic_onnx_fixture(tmp_path: Path) -> None:
    policy_path = tmp_path / "fixture.onnx"
    weight = numpy_helper.from_array(np.zeros((49, 12), dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
        "zero_go2_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 49])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 12])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, policy_path)
    policy_bytes = policy_path.read_bytes()
    manifest = DeploymentManifest(
        schema_version="motrix-deploy/v1",
        source=SourceSpec(framework="test", run_id="mujoco-smoke", checkpoint="fixture"),
        policy=PolicySpec(
            component_version="onnx/v1",
            payload_path="policy/model.onnx",
            sha256=sha256_bytes(policy_bytes),
            input=TensorSpec(name="obs", shape=(1, 49)),
            output=TensorSpec(name="actions", shape=(1, 12)),
        ),
        robot=_robot_spec(),
        task=_task_spec(),
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
    )
    artifact_path = tmp_path / "fixture.deploy"
    write_artifact(artifact_path, manifest, {"policy/model.onnx": policy_bytes})
    command = [
        sys.executable,
        "-c",
        "from motrix_deploy_tasks import main; raise SystemExit(main())",
        "sim2sim",
        f"artifact={artifact_path}",
        "command.velocity=[0.5,0.0,0.0]",
        "rollout.steps=5",
        "seed=1",
        "viewer=false",
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    duration_command = command.copy()
    steps_index = duration_command.index("rollout.steps=5")
    duration_command[steps_index : steps_index + 1] = ["rollout.steps=null", "rollout.duration_s=0.1"]
    duration = subprocess.run(duration_command, check=False, capture_output=True, text=True)
    invalid_bound = subprocess.run(
        [*command, "rollout.duration_s=0.1"],
        check=False,
        capture_output=True,
        text=True,
    )
    first_result = _rollout_result(first.stdout)
    second_result = _rollout_result(second.stdout)
    duration_result = _rollout_result(duration.stdout)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert duration.returncode == 0, duration.stderr
    assert invalid_bound.returncode == 2
    assert "Exactly one of rollout.steps and rollout.duration_s" in invalid_bound.stderr
    assert first_result["success"] is True
    assert first_result["completed_steps"] == 5
    assert first_result["trace_sha256"] == second_result["trace_sha256"]
    assert duration_result["completed_steps"] == 5
    assert "Converting 12 actuators from position servos to torque motors" in first.stdout


def _robot_spec() -> RobotSpec:
    return RobotSpec(
        base_link_name="base",
        joint_names=(
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ),
        default_joint_position=np.array(
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
            dtype=np.float32,
        ),
        position_lower=np.tile(np.array([-0.9472, -1.4, -2.6227], dtype=np.float32), 4),
        position_upper=np.tile(np.array([0.9472, 2.5, -0.84776], dtype=np.float32), 4),
        torque_limit=np.full(12, 24.0, dtype=np.float32),
    )


def _task_spec() -> TaskSpec:
    return TaskSpec(
        name="go2_walk/v1",
        observation_size=49,
        action_size=12,
        config={
            "action_scale": 0.25,
            "command_lower": [0.5, 0.0, 0.0],
            "command_upper": [0.5, 0.0, 0.0],
            "command_scale": [1.0, 1.0, 1.0],
            "feet_phase_offsets": [0.0, 0.5, 0.5, 0.0],
            "gait_frequency_hz": 2.0,
            "standing_threshold": 0.05,
            "kp": [35.0] * 12,
            "kd": [0.5] * 12,
            "raw_clip": [[-1.0] * 12, [1.0] * 12],
        },
    )
