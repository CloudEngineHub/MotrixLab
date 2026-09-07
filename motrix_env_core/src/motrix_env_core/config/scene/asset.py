# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene._utils import Vec2, Vec3, Vec4, optional_vec, resolve_path
from motrix_env_core.config.scene.base import SceneAssetCfg


@configclass(kw_only=True)
class TerrainGeneratorCfg(ABC):
    """Base config for build-time procedural height-field generation."""

    # Seed for deterministic terrain generation.
    seed: int = 0
    # Final vertical range of the normalized height field, in meters.
    height_scale: float = 0.05

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError(f"TerrainGeneratorCfg.seed must be non-negative, got {self.seed}")
        if not np.isfinite(self.height_scale) or self.height_scale < 0.0:
            raise ValueError(
                f"TerrainGeneratorCfg.height_scale must be finite and non-negative, got {self.height_scale}"
            )

    @abstractmethod
    def generate(self, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        """Generate unitless terrain samples as a MuJoCo-row-major 2D array."""


@configclass
class NoiseTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Independent uniformly distributed terrain-height noise."""

    # Reverse the generated row axis for parity with an equivalent image-backed height field.
    flip_y: bool = False

    def generate(self, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        del size
        rng = np.random.default_rng(self.seed)
        heights = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        return np.flipud(heights) if self.flip_y else heights


@configclass
class TextureCfg(SceneAssetCfg):
    """A file-backed 2D texture asset."""

    file: str | Path
    color_space: str = "srgb"
    gen_mipmaps: bool = True

    def validate(self, name: str) -> None:
        super().validate(name)
        path = resolve_path(self.file)
        if not path.is_file():
            raise FileNotFoundError(f"Texture file does not exist: {path}")
        if self.color_space not in ("srgb", "linear"):
            raise ValueError(f"TextureCfg.color_space must be 'srgb' or 'linear', got {self.color_space!r}")


@configclass
class SkyboxCfg(SceneAssetCfg):
    """A gradient skybox texture bound as the world's active skybox."""

    color_top: Vec3 = (0.4, 0.4, 0.4)
    color_bottom: Vec3 = (0.0, 0.0, 0.0)
    width: int = 512
    height: int = 3072
    color_space: str = "srgb"
    gen_mipmaps: bool = True

    def validate(self, name: str) -> None:
        super().validate(name)
        optional_vec("SkyboxCfg.color_top", self.color_top, 3)
        optional_vec("SkyboxCfg.color_bottom", self.color_bottom, 3)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"SkyboxCfg dimensions must be positive, got {(self.width, self.height)!r}")
        if self.color_space not in ("srgb", "linear"):
            raise ValueError(f"SkyboxCfg.color_space must be 'srgb' or 'linear', got {self.color_space!r}")


@configclass
class MaterialCfg(SceneAssetCfg):
    """A material asset with an optional reference to a configured texture."""

    texture: str | None = None
    color: Vec4 = (1.0, 1.0, 1.0, 1.0)
    texture_repeat: Vec2 = (1.0, 1.0)
    texture_uniform: bool = True
    metallic: float = 0.0
    roughness: float = 0.0

    def validate(self, name: str) -> None:
        super().validate(name)
        optional_vec("MaterialCfg.color", self.color, 4)
        optional_vec("MaterialCfg.texture_repeat", self.texture_repeat, 2)
        if self.texture == "":
            raise ValueError("MaterialCfg.texture must not be empty")


@configclass
class HFieldAssetCfg(SceneAssetCfg):
    """A file-backed height-field asset."""

    file: str | Path
    # Full world-space X/Y width. Backends convert this to their native half-extents.
    size: Vec2 = (16.0, 16.0)
    height_scale: float = 0.05

    def validate(self, name: str) -> None:
        super().validate(name)
        path = resolve_path(self.file)
        if not path.is_file():
            raise FileNotFoundError(f"HField file does not exist: {path}")
        optional_vec("HFieldAssetCfg.size", self.size, 2)


@configclass
class ProceduralHFieldAssetCfg(SceneAssetCfg):
    """An in-memory height-field asset generated while building the scene."""

    generator: TerrainGeneratorCfg
    # Full world-space X/Y width. Backends convert this to their native half-extents.
    size: Vec2 = (16.0, 16.0)
    shape: tuple[int, int] = (257, 257)

    def validate(self, name: str) -> None:
        super().validate(name)
        optional_vec("ProceduralHFieldAssetCfg.size", self.size, 2)
        if any(not np.isfinite(value) or value <= 0.0 for value in self.size):
            raise ValueError(f"ProceduralHFieldAssetCfg.size must be finite and positive, got {self.size!r}")
        if len(self.shape) != 2 or any(not isinstance(value, int) or value < 2 for value in self.shape):
            raise ValueError(
                f"ProceduralHFieldAssetCfg.shape must contain two dimensions of at least 2, got {self.shape!r}"
            )
        if not isinstance(self.generator, TerrainGeneratorCfg):
            raise TypeError(
                "ProceduralHFieldAssetCfg.generator must contain TerrainGeneratorCfg, "
                f"got {type(self.generator).__name__}"
            )
        self.generator.validate()
