# Config Overrides and Tuning

`HumanoidVelocityTrackingEnvCfg` provides defaults for the shared velocity-tracking logic. A robot task commonly
overrides selected subconfigs according to robot dimensions, joint ranges, actuator capability, and training behavior.
This page describes the override patterns and tuning considerations.

## 1. Override configs

Pass only the subconfigs that need to change when defining the flat-ground config. Omitted fields keep their defaults:

```python
return HumanoidVelocityTrackingEnvCfg(
    scene=scene,
    asset=asset,
    control_config=ControlCfg(action_scale=0.5),
    commands=CommandsCfg(...),
    normalization=NormalizationCfg(...),
    gait=GaitCfg(...),
    reward_config=RewardCfg(...),
    curriculum=CurriculumCfg(...),
)
```

## 2. `control_config`: action scale

The policy produces a joint-position residual around the default standing pose:

$$
q_{target}=q_{default}+action\_scale\cdot a
$$

`control_config.action_scale` controls how strongly a policy action changes the target joint angle. Action-space bounds
are derived from actuator control ranges, default angles, and `action_scale`.

```python
control_config = ControlCfg(action_scale=0.5)
```

## 3. `commands`: velocity distribution

`CommandsCfg` controls the sampled `[vx, vy, yaw_rate]` commands:

```python
commands = CommandsCfg(
    vel_limit=[
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    stand_prob=0.2,
    resampling_time=10.0,
)
```

| Field             | Meaning                                                                                  |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `vel_limit`       | Shape `(2, 3)`; the first row contains lower bounds and the second contains upper bounds |
| `stand_prob`      | Probability of sampling an all-zero standing command                                     |
| `resampling_time` | Command resampling interval in seconds                                                   |

Start with smaller translation and turning ranges. Expand them after the robot can stand and respond consistently.

## 4. `normalization`: observation scales and noise

`NormalizationCfg` defines observation scaling before values enter the networks, together with joint noise applied only
to actor observations. Each physical quantity is scaled before it is concatenated into the observation vector:

$$
x_{obs}=scale\cdot x_{raw}
$$

Scaling changes only the numerical input received by the actor and critic; it does not change simulation state,
velocity commands, or rewards. Increasing a scale amplifies changes in that observation component, while decreasing it
attenuates them. Scales should normally keep observations with different units within similar numerical ranges during
typical motion. This prevents large-valued inputs from dominating early optimization and small-valued inputs from being
difficult to use.

| Field           | Applies to                                                                        |
| --------------- | --------------------------------------------------------------------------------- |
| `base_lin_vel`  | Scale multiplier for body-frame linear velocity in the critic observation         |
| `base_ang_vel`  | Scale multiplier for body-frame angular velocity in actor and critic observations |
| `dof_pos`       | Scale multiplier for the joint-position residual from the default pose            |
| `dof_vel`       | Scale multiplier for joint velocity                                               |
| `noise_dof_pos` | Uniform actor-side joint-position noise amplitude                                 |
| `noise_dof_vel` | Uniform actor-side joint-velocity noise amplitude                                 |

Joint noise is added after scaling, so set `noise_dof_pos` and `noise_dof_vel` relative to the corresponding scaled
signals. The critic's privileged observation has no noise. Recheck these fields when joint units, velocity ranges, or
policy input scales change.

## 5. `gait`: biped gait reference

`GaitCfg` defines the shared two-foot periodic reference:

```python
gait = GaitCfg(
    period=1.0,
    swing_height=0.09,
    feet_phase_sigma=0.008,
)
```

| Field              | Meaning                                                            |
| ------------------ | ------------------------------------------------------------------ |
| `period`           | Gait-cycle duration in seconds                                     |
| `swing_height`     | Expected maximum swing-foot height above the local ground          |
| `feet_phase_sigma` | Scale that converts foot-height error into the `feet_phase` reward |

Choose `swing_height` according to leg length, sole-site placement, and terrain variation. Standing commands pin the
phase and do not require foot motion.

## 6. `reward_config`: rewards and pose regularization

`RewardCfg` contains shared reward scales and a small number of reward parameters:

```python
reward_config = RewardCfg(
    scales=RewardScales(
        tracking_lin_vel=...,
        tracking_ang_vel=...,
        penalty_action_rate=...,
        # Override other shared reward terms as needed.
    ),
    tracking_sigma=0.25,
    close_feet_threshold=0.15,
    pose_weights={
        "<joint-name>": 1.0,
        # Cover every joint on the robot body.
    },
)
```

| Field                  | Meaning                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| `scales`               | Scales for velocity tracking, stability, action, pose, and gait terms     |
| `tracking_sigma`       | Exponential-kernel scale for linear-velocity and yaw-rate tracking errors |
| `close_feet_threshold` | Lateral foot-distance threshold for the close-feet penalty                |
| `pose_weights`         | Relative penalty for each joint's deviation from its default pose         |

The `pose_weights` keys must exactly match all joint names on the robot body. Weights must be finite and non-negative.
Larger values hold a joint closer to its default pose; smaller values allow it to participate more in the gait.

## 7. `curriculum`: penalty curriculum

`CurriculumCfg` adjusts the overall scale of selected penalty terms according to the average length of completed
episodes:

| Field                     | Meaning                                                                  |
| ------------------------- | ------------------------------------------------------------------------ |
| `enabled`                 | Enable the curriculum; a disabled curriculum uses a fixed scale of `1.0` |
| `initial_scale`           | Penalty scale at the start of training                                   |
| `min_scale` / `max_scale` | Allowed scale range                                                      |
| `level_down_threshold`    | Decrease penalties below this average episode length                     |
| `level_up_threshold`      | Increase penalties above this average episode length                     |
| `degree`                  | Relative change applied at each update                                   |
| `penalty_terms`           | Reward terms affected by the curriculum                                  |

The defaults are a suitable starting point. If the robot cannot stand, first check the model, default pose, PD settings,
and termination conditions before relaxing penalties.

## 8. Terrain and spawn range

`spawn_xy_range` controls uniform reset-position sampling along X and Y. Flat-ground configs normally use `0.0`.

Flat and uneven-terrain configs should share robot, action, observation, reward, and termination settings. The
uneven-terrain config only needs to change the terrain scene and spawn range.

The height-field variant raises the robot base using local samples around its spawn point. Each step also measures foot
clearance relative to the local ground beneath each sole.
