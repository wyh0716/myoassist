# V3-B code archive

This branch preserves the source code and configuration used for the
MyoLegs22 + MotorArms V3-B experiment. Generated training sessions,
checkpoints, videos, figures, CSV files, and other result artifacts are not
stored in this branch.

## Authoritative training configuration

- `rl_train/train/train_configs/myolegs22_motor_arms_62obs_128_contact_smooth_V3b.json`
- MuJoCo model: `artifacts/myolegs22_planar_motor_arms_consistent_aux_CANDIDATE.xml`
- Reference input expected locally:
  `rl_train/reference_data/short_reference_gait_v3_contact_constrained_smooth_CANDIDATE.npz`

The reference NPZ is deliberately not committed because this archive is
limited to source and configuration. Its SHA-256 is:

```
72560c1b08aae607d9e6061279a3857d73f2aae340854784b534a37f0c5747a2
```

The XML SHA-256 is:

```
738e4b405a2fb033b710c4b2f94f29a0e73e30f744ec36b80a5f181774c13fd9
```

## V3-B contract

- 62-dimensional observation
- 22-dimensional learned lower-limb muscle action
- four arm motor commands appended inside the environment
- contralateral shoulder PD control and fixed 20-degree elbow PD target
- 0.5-second quintic PD target ramp after reset
- progress-plus-tracking forward reward
- 0.6-rad trajectory threshold with a 10-control-step grace window
- 50 Hz policy control and 2000 Hz MuJoCo physics
- PPO actor network `[128, 128]`

## Validation

From the repository root, with the reference NPZ restored at the path above:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/myoassist-mpl \
  .my_venv/bin/python3 -m pytest -p no:cacheprovider \
  tools/tests/test_motor_arms.py \
  tools/tests/test_motor_arms_training_interface.py \
  tools/tests/test_v3_motor_arms_environment.py \
  tools/tests/test_v3b_forward_and_termination.py -q
```

The archived code retains backward-compatible configuration defaults used by
older experiments. V3-C configuration, adaptive-reset tests, V3-C reference
data, and V3-C results are intentionally excluded from this branch.
