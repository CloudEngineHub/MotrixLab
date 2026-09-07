# Reward Function Design

The reward function tells the agent what behaviors are desired and is a core part of reinforcement learning environment design.

## Position of Reward Function in Training Loop

In MotrixLab's `DirectEnv` workflow, reward calculation occurs in the `compute_transition` phase of the `step` function:

```python
# Execution flow of DirectEnv.step()
def step(self, actions: np.ndarray) -> ArrayEnvState:
    # 1. Preparation phase: Clear rewards and state
    self._prev_physics_step()  # reward = 0.0, terminated = False, truncated = False

    # 2. Apply actions
    self._state = self.apply_action(actions, self._state)

    # 3. Physics simulation
    self.physics_step()  # Execute physics simulation

    # 4. Compute transition ← Reward function is calculated here
    self._state = self.compute_transition(self._state)

    # 5. Post-processing
    self._update_truncate()  # Check time truncation
    self._reset_done_envs()  # Reset completed environments
    self._state = self.compute_observation(self._state)  # Compute post-reset observations

    return self._state
```

You need to implement reward calculation logic in the `compute_transition` method of subclasses. For specific reward function design ideas, please refer to the training examples.

### Reward Component Design Principles

1. **Separation of Concerns**: Each reward function should handle a specific goal
2. **Weight Configuration**: Manage weights of different components through configuration files
3. **Normalization**: Keep reward values within reasonable ranges
4. **Smoothness**: Avoid hard thresholds, use exponential functions for smooth transitions

This approach makes reward functions modular, facilitating debugging and adjustment of individual component weights.

## Design Principles

1. **Clear Goal Orientation**: Reward functions should directly reflect task goals
2. **Reasonable Reward Range**: Avoid overly large or small reward values to maintain training stability
3. **Balance Exploration and Exploitation**: Appropriately reward behaviors close to goals, avoiding sparse rewards
4. **Avoid Reward Hacking**: Check if agents can obtain high rewards through unintended means
5. **Debug-Friendly**: Output reward decomposition information during development for optimization

By correctly implementing reward calculation in the `compute_transition` method, you can design effective learning signals for various robot tasks.
