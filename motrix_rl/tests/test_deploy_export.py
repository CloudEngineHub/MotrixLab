# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-agnostic deployment export service tests."""

import numpy as np
import pytest

from motrix_deploy.artifact import ControlSpec, TaskSpec
from motrix_deploy.contracts import RobotSpec
from motrix_deploy.profile import DeploymentProfile
from motrix_rl import checkpoints, runs
from motrix_rl.deploy.api import OnnxExportReport, OnnxModelArtifact, OnnxParityMetrics, PolicyTensorSpec
from motrix_rl.deploy.service import export_deploy_run


def _profile() -> DeploymentProfile:
    return DeploymentProfile(
        robot=RobotSpec(
            base_link_name="base",
            joint_names=("left", "right"),
            default_joint_position=np.zeros(2, dtype=np.float32),
            position_lower=np.full(2, -1.0, dtype=np.float32),
            position_upper=np.full(2, 1.0, dtype=np.float32),
            torque_limit=np.full(2, 3.0, dtype=np.float32),
        ),
        task=TaskSpec(
            name="test/v1",
            observation_size=4,
            action_size=2,
            config={},
        ),
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
    )


def test_deployment_export_injects_profile_builder_selected_by_run_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = runs.create_run_context(
        env_name="cartpole",
        rllib="skrl",
        train_backend="torch",
        algo="ppo",
        seed=1,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )
    checkpoint = run.checkpoint_dir / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    checkpoints.record_checkpoint_artifact(
        run.run_dir,
        checkpoints.BEST_POLICY,
        checkpoint,
        checkpoints.POLICY,
        checkpoint_format="pt",
    )
    model = OnnxModelArtifact(
        model_bytes=b"validated ONNX payload",
        report=OnnxExportReport(
            input_spec=PolicyTensorSpec(name="obs", shape=(None, 4), dtype="float32"),
            output_spec=PolicyTensorSpec(name="actions", shape=(None, 2), dtype="float32"),
            parity=OnnxParityMetrics(samples=8, max_abs_error=1e-6, max_rel_error=2e-6),
        ),
    )
    monkeypatch.setattr("motrix_rl.deploy.service.export_onnx_model", lambda *args, **kwargs: model)
    selected: list[str] = []

    result = export_deploy_run(
        run.run_dir,
        tmp_path / "test.deploy",
        profile_builder=lambda env_name: selected.append(env_name) or _profile(),
        validation_samples=8,
    )

    assert selected == ["cartpole"]
    assert result.artifact.manifest.task.name == "test/v1"
    assert result.artifact.manifest.source.framework == "skrl.ppo/torch"
    assert result.artifact.policy_path.read_bytes() == model.model_bytes
    assert result.validation_samples == 8
    assert not checkpoint.with_name("policy.onnx").exists()
