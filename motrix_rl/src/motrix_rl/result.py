# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from motrix_env_core.renderer import RenderConfig
from motrix_rl import checkpoints
from motrix_rl.runs import RunContext


@dataclass(frozen=True)
class TrainResult:
    """Framework-owned view of a completed training run."""

    run: RunContext

    @property
    def run_dir(self) -> Path:
        """Return the root directory of this training run."""
        return self.run.run_dir

    def find_best_policy(self) -> Path:
        """Find the best policy produced by this training result."""
        return checkpoints.best_policy(self.run.metadata, self.run.run_dir)

    def find_resume_checkpoint(self) -> Path:
        """Find the checkpoint that should be used to resume this training run."""
        return checkpoints.resume_checkpoint(self.run.run_dir, metadata=self.run.metadata)

    def play(
        self,
        *,
        cfg_override: dict | None = None,
        render: RenderConfig | None = RenderConfig(),
    ) -> None:
        """Play the best policy produced by this training run.

        Args:
            cfg_override: Optional provider-specific algorithm-config overrides for play.
            render: Interactive rendering, video recording, or ``None`` to disable rendering.
        """
        from motrix_rl import runner

        trainer = runner.create_run_handle(
            self.run,
            cfg_override=cfg_override,
            render=render,
        )
        trainer.play(str(self.find_best_policy()))
