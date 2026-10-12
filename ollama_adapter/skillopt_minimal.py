"""Minimal SkillOpt-style reproduction on official ALFWorld TextWorld tasks.

This intentionally reproduces the paper's core causal mechanism rather than
the complete framework: frozen actor -> trajectory reflection -> skill edit ->
held-out validation gate -> versioned accepted skill.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml


DEFAULT_SKILL = """# ALFWorld operating skill

1. Preserve the task goal throughout the episode.
2. Use `go to` to inspect likely receptacles before trying to take an object.
3. For look-under-light tasks: find and take the target, go to the named lamp,
   switch the lamp on if necessary, then examine the target while holding it.
4. For clean/heat/cool tasks: acquire the object, use the required appliance,
   then place it in the requested destination.
5. Avoid repeating an action when the observation did not improve.
6. Select the numeric ID of one action from the numbered admissible-action list.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434"))
    p.add_argument("--output-dir", default="/results/skillopt_minimal")
    p.add_argument("--train-indices", default="0", help="comma-separated ALFWorld train indices")
    p.add_argument("--validation-indices", default="0", help="comma-separated valid_seen indices")
    p.add_argument("--test-indices", default="0", help="comma-separated valid_unseen indices")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--action-retries", type=int, default=2, help="retries after an invalid model action")
    p.add_argument("--stagnation-window", type=int, default=8, help="stop after this many cyclic no-progress steps; 0 disables")
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-skill", help="optional UTF-8 Markdown skill file")
    p.add_argument("--skip-optimization", action="store_true")
    p.add_argument("--skip-no-skill-baseline", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="test bookkeeping without model/environment calls")
    return p.parse_args()


def comma_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "alfworld_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value

    config = expand(config)
    config["general"]["use_cuda"] = False
    config["env"]["type"] = "AlfredTWEnv"
    return config


def ollama_chat(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.time()
    response = requests.post(url.rstrip("/") + "/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json(), time.time() - started


def parse_json_content(body: dict[str, Any]) -> dict[str, Any]:
    content = str(body.get("message", {}).get("content", "")).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw": content}


def action_payload(model: str, prompt: str, seed: int) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "keep_alive": -1,
        "format": {
            "type": "object",
            "properties": {"action_id": {"type": "integer", "minimum": 0}},
            "required": ["action_id"],
        },
        "options": {"temperature": 0, "seed": seed, "num_ctx": 4096, "num_predict": 12},
    }


def edit_payload(model: str, prompt: str, seed: int) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "keep_alive": -1,
        "format": {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "string"},
                "candidate_skill": {"type": "string"},
                "edit_summary": {"type": "string"},
            },
            "required": ["diagnosis", "candidate_skill", "edit_summary"],
        },
        "options": {"temperature": 0.2, "seed": seed, "num_ctx": 8192, "num_predict": 500},
    }


def choose_action_id(raw: Any, admissible: list[str]) -> tuple[str, str, int | None]:
    """Map a compact model-selected index to an exact environment action."""
    try:
        if isinstance(raw, bool):
            raise ValueError
        action_id = int(str(raw).strip())
    except (TypeError, ValueError):
        return "", "invalid_action_id", None
    if 0 <= action_id < len(admissible):
        return admissible[action_id], "model_action_id", action_id
    return "", "action_id_out_of_range", action_id


def fallback_action(admissible: list[str], history: list[dict[str, Any]]) -> tuple[str, int]:
    """Prefer a state-changing action when every model retry is invalid."""
    recent = {item["selected_action"] for item in history[-3:]}
    prefixes = ("take ", "use ", "open ", "close ", "move ", "heat ", "cool ", "clean ", "go to ")
    for index, action in enumerate(admissible):
        if action not in recent and action.startswith(prefixes):
            return action, index
    for index, action in enumerate(admissible):
        if action not in recent:
            return action, index
    return admissible[0], 0


def cyclic_stagnation(history: list[dict[str, Any]], window: int) -> bool:
    """Detect a short action/observation cycle with no reward progress."""
    if window <= 0 or len(history) < window:
        return False
    recent = history[-window:]
    observations = {item["next_observation"] for item in recent}
    scores = {float(item["score"]) for item in recent}
    actions = {item["selected_action"] for item in recent}
    return len(observations) <= 2 and len(scores) == 1 and len(actions) <= 3


def blocked_for_loop(history: list[dict[str, Any]], observation: str, admissible: list[str]) -> list[str]:
    blocked: list[str] = []
    if len(history) >= 2:
        previous, before_previous = history[-1], history[-2]
        if (
            previous["selected_action"] == before_previous["selected_action"]
            and previous["next_observation"] == before_previous["next_observation"] == observation
        ):
            blocked.append(previous["selected_action"])
    if len(history) >= 4:
        recent = history[-4:]
        if len({item["next_observation"] for item in recent}) <= 2 and len({float(item["score"]) for item in recent}) == 1:
            for item in recent:
                action = item["selected_action"]
                if action.startswith(("examine ", "look", "inventory")):
                    blocked.append(action)
    unique = []
    for action in blocked:
        if action in admissible and action not in unique:
            unique.append(action)
    return unique if len(unique) < len(admissible) else []


def progress_summary(history: list[dict[str, Any]]) -> str:
    actions = [item["selected_action"] for item in history]
    held = next((action.removeprefix("take ").split(" from ", 1)[0] for action in reversed(actions) if action.startswith("take ")), "unknown")
    lamp_used = any(action.startswith("use ") and "lamp" in action for action in actions)
    return f"holding_or_last_taken={held}; lamp_used={lamp_used}; completed_actions={actions[-4:]}"


def make_actor_prompt(
    goal: str,
    observation: str,
    admissible: list[str],
    history: list[dict[str, Any]],
    skill: str,
    blocked_actions: list[str] | None = None,
) -> str:
    short_history = history[-4:]
    rendered_history = "\n".join(
        f"{item['step']}. action={item['selected_action']} -> observation={item['next_observation'][:350]}"
        for item in short_history
    ) or "(none)"
    blocked_actions = blocked_actions or []
    blocked_note = (
        "\nLOOP GUARD: recent actions made no progress. These actions are temporarily unavailable: "
        f"{json.dumps(blocked_actions, ensure_ascii=False)}\nChoose a state-changing action.\n"
        if blocked_actions else ""
    )
    numbered_actions = "\n".join(f"{index}: {action}" for index, action in enumerate(admissible))
    return f"""You are a frozen ALFWorld text-game agent. Follow the skill, but the environment's
admissible actions are authoritative. Choose exactly one admissible action.

TASK GOAL (never forget this):
{goal}

CURRENT SKILL:
{skill}

RECENT TRAJECTORY:
{rendered_history}

STRUCTURED PROGRESS:
{progress_summary(history)}

CURRENT OBSERVATION:
{observation}
{blocked_note}

NUMBERED ADMISSIBLE ACTIONS:
{numbered_actions}

Return only JSON with integer key `action_id`, for example {{"action_id": 2}}."""


@dataclass
class EpisodeResult:
    split: str
    task_index: int
    task_file: str
    goal: str
    skill_sha: str
    won: bool
    final_score: float
    steps: list[dict[str, Any]]
    invalid_action_count: int
    model_call_count: int
    latency_seconds: float
    done_reason: str


def stable_sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class AlfWorldRunner:
    def __init__(self, model: str, ollama_url: str, timeout: float, seed: int, action_retries: int = 2, stagnation_window: int = 8):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.seed = seed
        self.action_retries = action_retries
        self.stagnation_window = stagnation_window
        self.config = load_config()

    def run(self, split: str, task_index: int, skill: str, max_steps: int) -> EpisodeResult:
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        train_eval = {"train": "train", "valid_seen": "eval_in_distribution", "valid_unseen": "eval_out_of_distribution"}[split]
        random.seed(self.seed)
        np.random.seed(self.seed)
        alf_env = AlfredTWEnv(self.config, train_eval=train_eval)
        if not 0 <= task_index < len(alf_env.game_files):
            raise IndexError(f"task index {task_index} outside {split} size {len(alf_env.game_files)}")
        selected = str(alf_env.game_files[task_index])
        alf_env.game_files = [selected]
        alf_env.num_games = 1
        env = alf_env.init_env(batch_size=1)
        obs, info = env.reset()
        first_observation = str(obs[0])
        goal_match = re.search(r"Your task is to:\s*(.+)", first_observation, flags=re.IGNORECASE)
        goal = goal_match.group(1).strip() if goal_match else first_observation
        history: list[dict[str, Any]] = []
        invalid = 0
        final_score = 0.0
        won = False
        done_reason = "max_steps"

        for step in range(max_steps):
            if cyclic_stagnation(history, self.stagnation_window):
                done_reason = "cyclic_stagnation"
                break
            observation = str(obs[0])
            admissible = list(info["admissible_commands"][0])
            blocked = blocked_for_loop(history, observation, admissible)
            allowed = [action for action in admissible if action not in blocked]
            if not allowed:
                allowed = admissible
                blocked = []
            base_prompt = make_actor_prompt(goal, observation, allowed, history, skill, blocked)
            attempts: list[dict[str, Any]] = []
            latency = 0.0
            chosen = ""
            selection = ""
            parsed: dict[str, Any] = {}
            for attempt_index in range(self.action_retries + 1):
                retry_note = ""
                if attempts:
                    retry_note = (
                        "\n\nRETRY: Your previous action_id was not a valid integer index in the numbered list. "
                        f"Invalid output: {json.dumps(attempts[-1]['raw_action_id'], ensure_ascii=False)}. "
                        f"Return one integer from 0 through {len(allowed) - 1}."
                    )
                body, call_latency = ollama_chat(
                    self.ollama_url,
                    action_payload(self.model, base_prompt + retry_note, self.seed + attempt_index),
                    self.timeout,
                )
                latency += call_latency
                parsed = parse_json_content(body)
                raw_action_id = parsed.get("action_id", parsed.get("raw", ""))
                proposed, proposed_selection, proposed_id = choose_action_id(raw_action_id, allowed)
                is_valid = proposed_selection == "model_action_id"
                attempts.append({
                    "attempt": attempt_index,
                    "raw_action_id": raw_action_id,
                    "resolved_action_id": proposed_id,
                    "valid": is_valid,
                    "selection": proposed_selection,
                })
                if is_valid:
                    chosen, selection = proposed, proposed_selection
                    break
                invalid += 1
            if not chosen:
                chosen, fallback_id = fallback_action(allowed, history)
                selection = "heuristic_fallback_after_retries"
                parsed = {**parsed, "fallback_action_id": fallback_id}
            obs2, scores, dones, infos = env.step([chosen])
            final_score = float(scores[0]) if len(scores) else 0.0
            won = bool(infos.get("won", [False])[0]) if "won" in infos else final_score > 0
            record = {
                "step": step,
                "observation": observation,
                "admissible_actions": admissible,
                "model_output": parsed,
                "action_attempts": attempts,
                "selected_action": chosen,
                "selection": selection,
                "loop_guard_blocked": blocked,
                "next_observation": str(obs2[0]),
                "score": final_score,
                "done": bool(dones[0]),
                "latency_seconds": round(latency, 3),
            }
            history.append(record)
            obs, info = obs2, infos
            if dones[0]:
                done_reason = "environment_done"
                break

        model_calls = sum(len(item.get("action_attempts", [])) for item in history)
        latency_seconds = round(sum(float(item.get("latency_seconds", 0.0)) for item in history), 3)
        return EpisodeResult(
            split, task_index, selected, goal, stable_sha(skill), won, final_score,
            history, invalid, model_calls, latency_seconds, done_reason
        )


def compact_trajectories(episodes: list[EpisodeResult]) -> str:
    chunks = []
    for ep in episodes:
        actions = [s["selected_action"] for s in ep.steps]
        tail = ep.steps[-1]["next_observation"] if ep.steps else ""
        chunks.append(
            f"task={ep.task_index}; goal={ep.goal}; won={ep.won}; invalid={ep.invalid_action_count}; "
            f"actions={actions}; final_observation={tail[:500]}"
        )
    return "\n\n".join(chunks)


def propose_skill_edit(model: str, url: str, timeout: float, seed: int, skill: str, episodes: list[EpisodeResult]) -> dict[str, str]:
    prompt = f"""You optimize a reusable ALFWorld skill document for a frozen small language model.
Diagnose recurring trajectory failures and produce ONE bounded revision.

Rules:
- Preserve useful instructions.
- Add only reusable, actionable rules supported by trajectory evidence.
- Remove or replace rules that caused repetition or wrong ordering.
- Do not encode task IDs, room layouts, exact answers, or validation examples.
- Keep the entire candidate under 1800 characters.
- Return JSON with diagnosis, candidate_skill, and edit_summary.

CURRENT SKILL:
{skill}

TRAINING TRAJECTORIES:
{compact_trajectories(episodes)}"""
    body, _ = ollama_chat(url, edit_payload(model, prompt, seed), timeout)
    parsed = parse_json_content(body)
    candidate = str(parsed.get("candidate_skill", "")).strip()
    if not candidate or len(candidate) > 1800:
        return {"diagnosis": str(parsed.get("diagnosis", "invalid candidate")), "candidate_skill": skill, "edit_summary": "rejected_before_validation: invalid candidate"}
    return {
        "diagnosis": str(parsed.get("diagnosis", "")),
        "candidate_skill": candidate,
        "edit_summary": str(parsed.get("edit_summary", "")),
    }


def score(episodes: list[EpisodeResult]) -> dict[str, float]:
    if not episodes:
        return {
            "success_rate": 0.0, "mean_score": 0.0, "invalid_actions": 0.0,
            "mean_steps": 0.0, "model_calls": 0.0, "latency_seconds": 0.0,
        }
    return {
        "success_rate": sum(ep.won for ep in episodes) / len(episodes),
        "mean_score": sum(ep.final_score for ep in episodes) / len(episodes),
        "invalid_actions": float(sum(ep.invalid_action_count for ep in episodes)),
        "mean_steps": sum(len(ep.steps) for ep in episodes) / len(episodes),
        "model_calls": float(sum(ep.model_call_count for ep in episodes)),
        "latency_seconds": round(sum(ep.latency_seconds for ep in episodes), 3),
    }


def strictly_better(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    # Faithful to SkillOpt's held-out gate: only task reward can accept an edit.
    # Invalid-action and step statistics are diagnostics, never tie-breakers.
    return (candidate["success_rate"], candidate["mean_score"]) > (incumbent["success_rate"], incumbent["mean_score"])


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_set(runner: AlfWorldRunner, split: str, indices: list[int], skill: str, max_steps: int, output_dir: Path, label: str) -> list[EpisodeResult]:
    episodes = []
    for index in indices:
        print(f"[{label}] starting {split} task {index}", flush=True)
        ep = runner.run(split, index, skill, max_steps)
        episodes.append(ep)
        save_json(output_dir / "episodes" / f"{label}_{split}_{index}.json", asdict(ep))
        print(
            f"[{label}] finished task {index}: won={ep.won} steps={len(ep.steps)} "
            f"calls={ep.model_call_count} latency={ep.latency_seconds}s reason={ep.done_reason}",
            flush=True,
        )
    return episodes


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    skill = Path(args.initial_skill).read_text(encoding="utf-8") if args.initial_skill else DEFAULT_SKILL

    if args.dry_run:
        assert choose_action_id(1, ["inventory", "look"])[0] == "look"
        assert choose_action_id(2, ["inventory", "look"])[1] == "action_id_out_of_range"
        synthetic = [
            {"selected_action": "examine x", "next_observation": "same", "score": 0.0}
            for _ in range(8)
        ]
        assert cyclic_stagnation(synthetic, 8)
        assert blocked_for_loop(synthetic, "same", ["examine x", "take y"] ) == ["examine x"]
        assert fallback_action(["examine x", "take y"], synthetic)[0] == "take y"
        assert not strictly_better({"success_rate": 0.0, "mean_score": 0.0}, {"success_rate": 0.0, "mean_score": 0.0})
        (out / "best_skill.md").write_text(skill, encoding="utf-8")
        save_json(out / "dry_run.json", {"ok": True, "skill_sha": stable_sha(skill)})
        print(json.dumps({"dry_run": "ok", "output_dir": str(out)}))
        return 0

    data_dir = Path(os.environ.get("ALFWORLD_DATA", "/opt/alfworld_data"))
    if not data_dir.exists():
        raise SystemExit(f"ALFWORLD_DATA does not exist: {data_dir}")
    runner = AlfWorldRunner(
        args.model, args.ollama_url, args.timeout, args.seed, args.action_retries, args.stagnation_window
    )
    train_indices = comma_ints(args.train_indices)
    validation_indices = comma_ints(args.validation_indices)
    test_indices = comma_ints(args.test_indices)

    train_eps = run_set(runner, "train", train_indices, skill, args.max_steps, out, "train_static")
    edit = {"diagnosis": "optimization skipped", "candidate_skill": skill, "edit_summary": "none"}
    gate: dict[str, Any] = {"accepted": False, "reason": "optimization skipped"}
    if not args.skip_optimization:
        edit = propose_skill_edit(args.model, args.ollama_url, args.timeout, args.seed, skill, train_eps)
        candidate = edit["candidate_skill"]
        incumbent_eps = run_set(runner, "valid_seen", validation_indices, skill, args.max_steps, out, "validation_incumbent")
        candidate_eps = run_set(runner, "valid_seen", validation_indices, candidate, args.max_steps, out, "validation_candidate")
        incumbent_score = score(incumbent_eps)
        candidate_score = score(candidate_eps)
        accepted = stable_sha(candidate) != stable_sha(skill) and strictly_better(candidate_score, incumbent_score)
        gate = {
            "accepted": accepted,
            "incumbent_skill_sha": stable_sha(skill),
            "candidate_skill_sha": stable_sha(candidate),
            "incumbent": incumbent_score,
            "candidate": candidate_score,
            "criterion": "strict lexicographic improvement in (success_rate, mean_score)",
        }
        if accepted:
            skill = candidate

    no_skill_eps: list[EpisodeResult] = []
    if not args.skip_no_skill_baseline:
        no_skill_eps = run_set(runner, "valid_unseen", test_indices, "", args.max_steps, out, "test_no_skill")
    test_eps = run_set(runner, "valid_unseen", test_indices, skill, args.max_steps, out, "test_best")
    (out / "best_skill.md").write_text(skill, encoding="utf-8")
    save_json(out / "candidate_edit.json", edit)
    save_json(out / "validation_gate.json", gate)
    summary = {
        "model": args.model,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "action_retries": args.action_retries,
        "stagnation_window": args.stagnation_window,
        "train": score(train_eps),
        "validation_gate": gate,
        "test_no_skill": score(no_skill_eps) if no_skill_eps else None,
        "test": score(test_eps),
        "best_skill_sha": stable_sha(skill),
        "scope": "minimal first-week SkillOpt core-loop reproduction; not full-paper reproduction",
    }
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
