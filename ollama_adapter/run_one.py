"""Run one official ALFWorld TextWorld task with a local Ollama model."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434"))
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--output", default="/results/alfworld_one.json")
    p.add_argument("--split", choices=("train", "valid_seen", "valid_unseen"), default="train")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--prepare-only", action="store_true", help="write one task prompt without calling Ollama")
    p.add_argument("--model-response-input", help="read a prepared Ollama JSON answer and submit one action")
    return p.parse_args()


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "alfworld_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # ALFWorld's config loader expands environment variables in the official
    # YAML.  Do the same for this small standalone adapter.
    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value
    return expand(config)


def make_ollama_payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "format": {
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
        },
        "options": {"temperature": 0, "seed": 0, "num_ctx": 2048, "num_predict": 32},
    }


def ollama_json(url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    payload = make_ollama_payload(model, prompt)
    response = requests.post(url.rstrip("/") + "/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    content = body.get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Keep the adapter tolerant of models/servers that ignore JSON schema.
        match = re.search(r'"action"\s*:\s*"([^"]+)"', content)
        if match:
            return {"action": match.group(1), "reasoning": content}
        return {"action": content.strip(), "reasoning": content}


def choose_action(raw: str, admissible: list[str]) -> tuple[str, str]:
    # Exact matching is required by TextWorld.  Accept only a verbatim action
    # from the environment; a fallback keeps a connectivity test safe.
    candidate = raw.strip()
    for action in admissible:
        if candidate == action:
            return action, "model_exact"
    lowered = candidate.lower()
    for action in admissible:
        if action.lower() == lowered:
            return action, "model_case_normalized"
    return admissible[0], "fallback_first_admissible"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("ALFWORLD_DATA", "/opt/alfworld_data")
    data_dir = Path(os.environ["ALFWORLD_DATA"])
    if not data_dir.exists():
        raise SystemExit(f"ALFWORLD_DATA does not exist: {data_dir}. Run download_alfworld_data.py first.")

    # Import after ALFWORLD_DATA is set so ALFWorld/TextWorld see the same path.
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    config = load_config()
    config["general"]["use_cuda"] = False
    config["env"]["type"] = "AlfredTWEnv"

    train_eval = {
        "train": "train",
        "valid_seen": "eval_in_distribution",
        "valid_unseen": "eval_out_of_distribution",
    }[args.split]

    random.seed(0)
    np.random.seed(0)
    alf_env = AlfredTWEnv(config, train_eval=train_eval)
    if not alf_env.game_files:
        raise SystemExit(f"No solvable ALFWorld games found for split={args.split}")
    if not 0 <= args.task_index < len(alf_env.game_files):
        raise SystemExit(
            f"task-index {args.task_index} is out of range for split={args.split}; "
            f"available games: {len(alf_env.game_files)}"
        )
    selected = str(alf_env.game_files[args.task_index])
    # Register only the selected official game.  This makes task-index fully
    # deterministic and avoids relying on TextWorld's internal episode order.
    alf_env.game_files = [selected]
    alf_env.num_games = 1
    env = alf_env.init_env(batch_size=1)
    obs, info = env.reset()

    first_admissible = list(info["admissible_commands"][0])
    first_observation = obs[0] if isinstance(obs, (list, tuple, np.ndarray)) else obs
    first_prompt = (
        "You are an ALFWorld text-game agent. Choose exactly one action from the admissible list.\n"
        "Return only JSON with key action. Copy the action verbatim; do not explain.\n\n"
        f"OBSERVATION:\n{first_observation}\n\n"
        f"ADMISSIBLE ACTIONS:\n{json.dumps(first_admissible, ensure_ascii=False)}"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        prepared = {
            "benchmark": "ALFWorld",
            "mode": "official_textworld_prepare",
            "split": args.split,
            "task_index": args.task_index,
            "task_file": selected,
            "observation": str(first_observation),
            "admissible_actions": first_admissible,
            "prompt": first_prompt,
            "ollama_url": args.ollama_url,
            "ollama_payload": make_ollama_payload(args.model, first_prompt),
        }
        output.write_text(json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
        return 0

    supplied_model_obj: dict[str, Any] | None = None
    if args.model_response_input:
        if args.max_steps != 1:
            raise SystemExit("--model-response-input currently supports --max-steps 1 only")
        supplied_model_obj = json.loads(Path(args.model_response_input).read_text(encoding="utf-8-sig"))

    records: list[dict[str, Any]] = []
    done_reason = "max_steps"
    model_calls = 0
    final_score = 0.0
    won = False
    for step in range(args.max_steps):
        admissible = list(info["admissible_commands"][0])
        observation = obs[0] if isinstance(obs, (list, tuple, np.ndarray)) else obs
        task_prompt = (
            "You are an ALFWorld text-game agent. Choose exactly one action from the admissible list.\n"
            "Return only JSON with key action. Copy the action verbatim; do not explain.\n\n"
            f"OBSERVATION:\n{observation}\n\n"
            f"ADMISSIBLE ACTIONS:\n{json.dumps(admissible, ensure_ascii=False)}"
        )
        started = time.time()
        model_obj = supplied_model_obj if supplied_model_obj is not None else ollama_json(
            args.ollama_url, args.model, task_prompt, args.timeout
        )
        model_calls += 1
        chosen, selection = choose_action(str(model_obj.get("action", "")), admissible)
        obs2, scores, dones, infos = env.step([chosen])
        score = float(scores[0]) if len(scores) else 0.0
        final_score = score
        won = bool(infos.get("won", [False])[0]) if isinstance(infos, dict) and "won" in infos else bool(score > 0)
        records.append({
            "step": step,
            "observation": str(observation),
            "admissible_count": len(admissible),
            "model_output": model_obj,
            "selected_action": chosen,
            "selection": selection,
            "score": score,
            "done": bool(dones[0]),
            "latency_seconds": round(time.time() - started, 3),
        })
        obs, info = obs2, infos
        if dones[0]:
            done_reason = "environment_done"
            break

    result = {
        "benchmark": "ALFWorld",
        "mode": "official_textworld",
        "split": args.split,
        "task_index": args.task_index,
        "task_file": selected,
        "model": args.model,
        "ollama_url": args.ollama_url,
        "model_calls": model_calls,
        "num_steps": len(records),
        "won": won,
        "final_score": final_score,
        "done_reason": done_reason,
        "steps": records,
        "scope": "one ALFWorld task; official TextWorld environment + Ollama adapter, not a full benchmark score",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
