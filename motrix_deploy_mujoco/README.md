# motrix-deploy-mujoco

`motrix-deploy-mujoco` is the MuJoCo backend plugin for `motrix-deploy`. It discovers a registered MotrixLab
environment configuration, builds its `SceneCfg` through `MuJoCoSceneCompiler` with deployment-owned MuJoCo timestep and
solver settings, converts the policy-training position actuators to deployment torque motors, and implements the
`RobotInterface` lifecycle.

The plugin is advertised through the `motrix_deploy.backends` Python entry-point group and is loaded only when a
deployment recipe selects `backend.name=mujoco`.

The workspace provides Go2 recipes for both registered scenes:

- `configs/deploy/sim2sim/go2_walk_sim2sim.yaml` selects `go2-walk-rough` and remains the CLI default.
- `configs/deploy/sim2sim/go2_walk_flat_sim2sim.yaml` selects `go2-walk-flat` explicitly.

Both use a `0.002` second MuJoCo timestep and `100` solver iterations. The flat recipe resets the base at `z=0.331` to keep
the default feet clear of the plane.
