# ALFWorld Skill Manager phase-2 experiment

This experiment extends the week-1 single-update reproduction without
changing that baseline. It evaluates a two-level design:

- **Actor (inner loop):** frozen `Qwen/Qwen3.5-4B` policy loaded directly with
  Hugging Face Transformers and selecting an admissible
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

## Native Linux (without Docker or Ollama)

The recommended cloud environment is Ubuntu 22.04, Python 3.12, PyTorch 2.8,
and CUDA 12.8. Install the native ALFWorld build dependencies and the direct
Transformers backend from the project root:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git unzip
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.qwen35.txt
```

Validate bookkeeping without loading model weights:

```bash
python ollama_adapter/skill_manager_experiment.py --dry-run
```

Run the complete fixed 5/5/5 experiment. The first run downloads the model and
the official ALFWorld text data:

```bash
chmod +x run_skill_manager_small.sh
./run_skill_manager_small.sh \
  --rounds 3 \
  --patience 2 \
  --max-steps 20 \
  --action-retries 2 \
  --stagnation-window 8 \
  --min-efficiency-gain 0.10
```

The launcher uses `python`, local data under `./data`, local results under
`./results`, and FP16 inference. Override these with the `PYTHON_BIN`,
`ALFWORLD_DATA`, `MODEL`, `DEVICE_MAP`, and `MODEL_DTYPE` environment variables.
For a previously downloaded model, set `MODEL` to its local directory.
