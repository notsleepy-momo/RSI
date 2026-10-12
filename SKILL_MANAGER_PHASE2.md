# ALFWorld Skill Manager phase-2 experiment

This experiment extends the week-1 single-update reproduction without
changing that baseline. It evaluates a two-level design:

- **Actor (inner loop):** frozen `qwen3.5:4b` policy selecting an admissible
  ALFWorld action.
- **Skill Manager (outer loop):** collects training trajectories, proposes a
  bounded skill edit, validates it against the incumbent, and versions only
  accepted edits.

## Fixed small split

`configs/splits_small_5.json` fixes five train, five `valid_seen`, and five
`valid_unseen` indices. Test indices are never used for skill updates.

With the default three rounds, training batches are:

1. `0,3`
2. `1,4`
3. `2`

Every candidate is compared with its incumbent on all five validation tasks.

## Acceptance gate

An edit must not reduce validation success rate or mean environment score and
must not increase invalid actions. It is accepted when either reward improves,
or reward ties and at least one stable policy-efficiency metric improves by
10%: mean steps, model calls, or invalid actions. Wall-clock latency is saved
as a diagnostic but is excluded from acceptance because it depends on machine
load.

## Comparisons

After all updates, the untouched test set is evaluated with:

1. no skill;
2. the static initial skill;
3. the final managed skill.

The output directory contains `summary.json`, `update_history.json`, per-round
gate records, per-episode trajectories, and a versioned `skills/` directory.

## Commands

Validate bookkeeping and container packaging without model calls:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\run_skill_manager_small.ps1 `
  -DryRun
```

Run the complete fixed 5/5/5 experiment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\run_skill_manager_small.ps1 `
  -Model qwen3.5:4b `
  -Rounds 3 `
  -Patience 2 `
  -MaxSteps 20 `
  -ActionRetries 2 `
  -StagnationWindow 8 `
  -MinEfficiencyGain 0.10
```

The full run is intentionally not part of the dry-run verification. On the
current machine, one 20-step episode can take roughly 25 minutes, so reserve a
multi-hour uninterrupted window for the complete experiment.
