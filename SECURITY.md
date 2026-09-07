# Security policy

Please do not report security vulnerabilities in a public issue. Use GitHub's
private vulnerability reporting when enabled, or email
`developers@motphys.com` with the subject `MotrixLab security report`.
Include the affected version/commit, reproduction steps, impact, and a safe
way to contact you. Do not include credentials or private robot-network data
in the report.

The maintainers will acknowledge a report within 10 business days and will
coordinate a fix and disclosure timeline with the reporter. This policy covers
the MotrixLab source and its CI configuration; vulnerabilities in MotrixSim,
MuJoCo, PyTorch, the Unitree SDK, or other dependencies should also be
reported to their upstream maintainers.

## Supported versions

| Version or branch | Security fixes |
| --- | --- |
| Latest stable release (`vMAJOR.MINOR.PATCH`) and its `stable` branch | Yes |
| `main` | Best effort; fixes land here before the next stable release |
| Older tags | No; upgrade to the latest stable release |

## Hardware safety

The Unitree deployment code can send commands to physical robots. Treat it as
high-risk software: use a suspended robot, a clear workspace, an independent
emergency stop, and the hardware checklist in the deployment documentation.
Never use a public issue or pull request to request an unsafe live-robot test.
