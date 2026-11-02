# Settings

One setting per method and benchmark, selected near the forgetting target.

## TOFU, forget 1%

Forgetting target: 0.408.

| Method | Settings | Forgetting | Utility |
|---|---|:-:|:-:|
| PINT | `lambda=300 lr=1e-5` | 0.417 | 0.601 |
| RMU | `steering_coeff=50 alpha=1.0 gamma=1.0 lr=1e-5` | 0.403 | 0.535 |
| NPO | `alpha=5 beta=0.5 lr=1e-5` | 0.427 | 0.598 |
| SimNPO | `beta=4.5 gamma=0.25 delta=1.0 lr=2e-5` | 0.385 | 0.588 |
| UNDIAL | `alpha=2 beta=30 lr=1e-5` | 0.403 | 0.609 |
| GradDiff | `lr=2e-5 alpha=10` | 0.411 | 0.585 |
| GradAscent | `lr=1e-5` | 0.388 | 0.587 |

```bash
bash scripts/unlearn.sh tofu pint forget_split=forget01 trainer.method_args.lambda=300 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh tofu rmu forget_split=forget01 trainer.method_args.steering_coeff=50 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh tofu npo forget_split=forget01 trainer.method_args.alpha=5 trainer.method_args.beta=0.5 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh tofu simnpo forget_split=forget01 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.25 trainer.method_args.delta=1.0 trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh tofu undial forget_split=forget01 trainer.method_args.alpha=2 trainer.method_args.beta=30 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh tofu graddiff forget_split=forget01 trainer.args.learning_rate=2e-5 trainer.method_args.alpha=10
bash scripts/unlearn.sh tofu gradascent forget_split=forget01 trainer.args.learning_rate=1e-5
```

## TOFU, forget 5%

Forgetting target: 0.384.

| Method | Settings | Forgetting | Utility |
|---|---|:-:|:-:|
| PINT | `lambda=450 lr=2e-5` | 0.381 | 0.597 |
| RMU | `steering_coeff=1 alpha=1.0 gamma=1.0 lr=1e-5` | 0.367 | 0.576 |
| NPO | `alpha=5 beta=0.1 lr=1e-5` | 0.370 | 0.553 |
| SimNPO | `beta=3.5 gamma=0.125 delta=1.0 lr=2e-5` | 0.385 | 0.593 |
| UNDIAL | `alpha=2 beta=3 lr=3e-4` | 0.401 | 0.113 |
| GradDiff | `lr=3e-5 alpha=10` | 0.373 | 0.577 |
| GradAscent | `lr=5e-6` | 0.477 | 0.546 |

```bash
bash scripts/unlearn.sh tofu pint forget_split=forget05 trainer.method_args.lambda=450 trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh tofu rmu forget_split=forget05 trainer.method_args.steering_coeff=1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh tofu npo forget_split=forget05 trainer.method_args.alpha=5 trainer.method_args.beta=0.1 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh tofu simnpo forget_split=forget05 trainer.method_args.beta=3.5 trainer.method_args.gamma=0.125 trainer.method_args.delta=1.0 trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh tofu undial forget_split=forget05 trainer.method_args.alpha=2 trainer.method_args.beta=3 trainer.args.learning_rate=3e-4
bash scripts/unlearn.sh tofu graddiff forget_split=forget05 trainer.args.learning_rate=3e-5 trainer.method_args.alpha=10
bash scripts/unlearn.sh tofu gradascent forget_split=forget05 trainer.args.learning_rate=5e-6
```

## TOFU, forget 10%

Forgetting target: 0.381.

| Method | Settings | Forgetting | Utility |
|---|---|:-:|:-:|
| PINT | `lambda=380 lr=2e-5` | 0.378 | 0.597 |
| RMU | `steering_coeff=5 alpha=1.0 gamma=1.0 lr=7.5e-6` | 0.364 | 0.557 |
| NPO | `alpha=2 beta=0.5 lr=5e-5` | 0.393 | 0.582 |
| SimNPO | `beta=4.5 gamma=0.25 delta=1.0 lr=2e-5` | 0.373 | 0.587 |
| UNDIAL | `alpha=5 beta=10 lr=1e-4` | 0.360 | 0.496 |
| GradDiff | `lr=2.5e-5 alpha=9` | 0.381 | 0.586 |
| GradAscent | `lr=5e-6` | 0.029 | 0.000 |

```bash
bash scripts/unlearn.sh tofu pint forget_split=forget10 trainer.method_args.lambda=380 trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh tofu rmu forget_split=forget10 trainer.method_args.steering_coeff=5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=7.5e-6
bash scripts/unlearn.sh tofu npo forget_split=forget10 trainer.method_args.alpha=2 trainer.method_args.beta=0.5 trainer.args.learning_rate=5e-5
bash scripts/unlearn.sh tofu simnpo forget_split=forget10 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.25 trainer.method_args.delta=1.0 trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh tofu undial forget_split=forget10 trainer.method_args.alpha=5 trainer.method_args.beta=10 trainer.args.learning_rate=1e-4
bash scripts/unlearn.sh tofu graddiff forget_split=forget10 trainer.args.learning_rate=2.5e-5 trainer.method_args.alpha=9
bash scripts/unlearn.sh tofu gradascent forget_split=forget10 trainer.args.learning_rate=5e-6
```

## MUSE News

Forgetting target: 0.3243.

| Method | Settings | Forgetting | Utility |
|---|---|:-:|:-:|
| PINT | `lambda=14000 lr=1e-5` | 0.330 | 0.304 |
| RMU | `steering_coeff=10 alpha=1.0 gamma=1.0 lr=5e-5` | 0.329 | 0.330 |
| NPO | `beta=0.01 retain_loss_type=NLL alpha=1.0 gamma=1.0 lr=2e-5` | 0.315 | 0.283 |
| SimNPO | `beta=1.0 gamma=0.25 delta=1.0 alpha=1.0 lr=1e-4` | 0.344 | 0.349 |
| UNDIAL | `beta=32 retain_loss_type=NLL alpha=1.0 gamma=1.0 lr=1e-4` | 0.315 | 0.302 |
| GradDiff | `alpha=1.0 gamma=1.0 retain_loss_type=NLL lr=1e-5` | 0.326 | 0.326 |
| GradAscent | `lr=9.8e-7` | 0.302 | 0.173 |

```bash
bash scripts/unlearn.sh muse pint data_split=News trainer.method_args.lambda=14000 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh muse rmu data_split=News trainer.method_args.steering_coeff=10 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=5e-5
bash scripts/unlearn.sh muse npo data_split=News trainer.method_args.beta=0.01 trainer.method_args.retain_loss_type=NLL trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh muse simnpo data_split=News trainer.method_args.beta=1.0 trainer.method_args.gamma=0.25 trainer.method_args.delta=1.0 trainer.method_args.alpha=1.0 trainer.args.learning_rate=1e-4
bash scripts/unlearn.sh muse undial data_split=News trainer.method_args.beta=32 trainer.method_args.retain_loss_type=NLL trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=1e-4
bash scripts/unlearn.sh muse graddiff data_split=News trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh muse gradascent data_split=News trainer.args.learning_rate=9.8e-7
```

## MUSE Books

Forgetting target: 0.3031.

| Method | Settings | Forgetting | Utility |
|---|---|:-:|:-:|
| PINT | `lambda=20 lr=2e-5` | 0.303 | 0.671 |
| RMU | `steering_coeff=5 alpha=1.0 gamma=1.0 lr=2.1e-5` | 0.296 | 0.634 |
| NPO | `lr=1e-5` | 0.306 | 0.594 |
| SimNPO | `beta=1.5 gamma=0.25 delta=1.0 alpha=1.0 lr=1e-5` | 0.300 | 0.599 |
| UNDIAL | `beta=64 retain_loss_type=NLL alpha=1.0 gamma=1.0 lr=8e-5` | 0.304 | 0.578 |
| GradDiff | `alpha=1.0 gamma=1.0 retain_loss_type=NLL lr=6e-7` | 0.302 | 0.558 |
| GradAscent | `lr=6e-7` | 0.281 | 0.532 |

```bash
bash scripts/unlearn.sh muse pint data_split=Books trainer.method_args.lambda=20 trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh muse rmu data_split=Books trainer.method_args.steering_coeff=5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=2.1e-5
bash scripts/unlearn.sh muse npo data_split=Books trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh muse simnpo data_split=Books trainer.method_args.beta=1.5 trainer.method_args.gamma=0.25 trainer.method_args.delta=1.0 trainer.method_args.alpha=1.0 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh muse undial data_split=Books trainer.method_args.beta=64 trainer.method_args.retain_loss_type=NLL trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=8e-5
bash scripts/unlearn.sh muse graddiff data_split=Books trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL trainer.args.learning_rate=6e-7
bash scripts/unlearn.sh muse gradascent data_split=Books trainer.args.learning_rate=6e-7
```

## WMDP Cyber

Forgetting target: 0.25.

| Method | Settings | Forgetting | Utility |
|---|---|:-:|:-:|
| PINT | `lambda=16000 lr=5e-5` | 0.263 | 0.573 |
| RMU | `steering_coeff=0.5 alpha=1.0 gamma=1.0 lr=5e-5` | 0.247 | 0.568 |
| NPO | `lr=1e-4` | 0.246 | 0.255 |
| SimNPO | `beta=1.0 gamma=0.25 delta=1.0 alpha=1.0 lr=2e-5` | 0.242 | 0.285 |
| UNDIAL | `beta=0.25 retain_loss_type=NLL alpha=1.0 gamma=1.0 lr=1e-4` | 0.246 | 0.255 |
| GradDiff | `alpha=1.0 gamma=1.0 retain_loss_type=NLL lr=5e-6` | 0.247 | 0.544 |
| GradAscent | `lr=3e-5` | 0.243 | 0.247 |

```bash
bash scripts/unlearn.sh wmdp pint data_split=cyber trainer.method_args.lambda=16000 trainer.args.learning_rate=5e-5
bash scripts/unlearn.sh wmdp rmu data_split=cyber trainer.method_args.steering_coeff=0.5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=5e-5
bash scripts/unlearn.sh wmdp npo data_split=cyber trainer.args.learning_rate=1e-4
bash scripts/unlearn.sh wmdp simnpo data_split=cyber trainer.method_args.beta=1.0 trainer.method_args.gamma=0.25 trainer.method_args.delta=1.0 trainer.method_args.alpha=1.0 trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh wmdp undial data_split=cyber trainer.method_args.beta=0.25 trainer.method_args.retain_loss_type=NLL trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=1e-4
bash scripts/unlearn.sh wmdp graddiff data_split=cyber trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL trainer.args.learning_rate=5e-6
bash scripts/unlearn.sh wmdp gradascent data_split=cyber trainer.args.learning_rate=3e-5
```

## WMDP Bio

Forgetting target: 0.25.

| Method | Settings | Forgetting | Utility |
|---|---|:-:|:-:|
| PINT | `lambda=30000 lr=5e-5` | 0.268 | 0.566 |
| RMU | `steering_coeff=2 alpha=1.0 gamma=1.0 lr=5e-5` | 0.268 | 0.519 |
| NPO | `beta=2.5 alpha=5 gamma=1.0 retain_loss_type=NLL lr=2e-5` | 0.257 | 0.278 |
| SimNPO | `beta=4.5 gamma=1.0 delta=0.0 alpha=1.0 lr=4e-5` | 0.247 | 0.229 |
| UNDIAL | `beta=0.25 retain_loss_type=NLL alpha=1.0 gamma=1.0 lr=4.5e-5` | 0.266 | 0.255 |
| GradDiff | `alpha=1.0 gamma=1.0 retain_loss_type=NLL lr=2.1e-6` | 0.262 | 0.571 |
| GradAscent | `lr=1.8e-7` | 0.255 | 0.396 |

```bash
bash scripts/unlearn.sh wmdp pint data_split=bio trainer.method_args.lambda=30000 trainer.args.learning_rate=5e-5
bash scripts/unlearn.sh wmdp rmu data_split=bio trainer.method_args.steering_coeff=2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=5e-5
bash scripts/unlearn.sh wmdp npo data_split=bio trainer.method_args.beta=2.5 trainer.method_args.alpha=5 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL trainer.args.learning_rate=2e-5
bash scripts/unlearn.sh wmdp simnpo data_split=bio trainer.method_args.beta=4.5 trainer.method_args.gamma=1.0 trainer.method_args.delta=0.0 trainer.method_args.alpha=1.0 trainer.args.learning_rate=4e-5
bash scripts/unlearn.sh wmdp undial data_split=bio trainer.method_args.beta=0.25 trainer.method_args.retain_loss_type=NLL trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.args.learning_rate=4.5e-5
bash scripts/unlearn.sh wmdp graddiff data_split=bio trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL trainer.args.learning_rate=2.1e-6
bash scripts/unlearn.sh wmdp gradascent data_split=bio trainer.args.learning_rate=1.8e-7
```

## Gaussian poisons

| Method | Settings | Checkpoint | Detectability | Utility |
|---|---|:-:|:-:|:-:|
| PINT | `lambda=100 lr=3e-5` | 150 | -0.008 | 0.914 |
| RMU | `steering_coeff=10 lr=3e-5` | 75 | 0.039 | 0.370 |
| NPO | `beta=1e-4 lr=3e-5` | 100 | -0.072 | 0.000 |
| SimNPO | `beta=0.2 lr=1e-5` | 150 | -0.013 | 0.631 |
| UNDIAL | `beta=30 lr=1e-4` | 200 | 0.475 | 0.471 |
| GradDiff | `alpha=1 lr=1e-5` | 200 | 0.176 | 0.000 |
| GradAscent | `lr=7e-6` | 125 | 0.065 | 0.000 |

```bash
bash scripts/unlearn.sh olmo_gw pint trainer.method_args.lambda=100 trainer.args.learning_rate=3e-5
bash scripts/unlearn.sh olmo_gw rmu trainer.method_args.steering_coeff=10 trainer.args.learning_rate=3e-5
bash scripts/unlearn.sh olmo_gw npo trainer.method_args.beta=1e-4 trainer.args.learning_rate=3e-5
bash scripts/unlearn.sh olmo_gw simnpo trainer.method_args.beta=0.2 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh olmo_gw undial trainer.method_args.beta=30 trainer.args.learning_rate=1e-4
bash scripts/unlearn.sh olmo_gw graddiff trainer.method_args.alpha=1 trainer.args.learning_rate=1e-5
bash scripts/unlearn.sh olmo_gw gradascent trainer.args.learning_rate=7e-6
```

## bioS

```bash
bash scripts/unlearn.sh bios pint bios_seed=0 trainer.args.seed=1
bash scripts/unlearn.sh bios rmu bios_seed=0 trainer.args.seed=1
```

## Llama-3-70B, WMDP Cyber

WMDP Cyber accuracy 0.276, MMLU 0.750 (mean over seeds 1, 2 and 3).

```bash
bash scripts/unlearn.sh wmdp pint_70b trainer.args.seed=1
```
