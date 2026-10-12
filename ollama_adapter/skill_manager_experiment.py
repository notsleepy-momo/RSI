"""Multi-round, two-level Skill Manager experiment for ALFWorld.

The frozen Actor comes from skillopt_minimal.  This module owns the outer
loop: collect training evidence, propose one bounded skill edit, compare it
against the incumbent on a fixed validation set, version accepted skills, and
evaluate no-skill/static/managed policies on an untouched test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from skillopt_minimal import (
    DEFAULT_SKILL,
    AlfWorldRunner,
    EpisodeResult,
    propose_skill_edit,
    run_set,
    save_json,
    score,
    stable_sha,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--ollama-url", default="http://host.docker.internal:11434")
    p.add_argument("--output-dir", default="/results/skill_manager_small")
    p.add_argument("--split-manifest", default="/opt/alfworld_project/configs/splits_small_5.json")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--patience", type=int, default=2, help="stop after this many consecutive rejected rounds; 0 disables")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--action-retries", type=int, default=2)
    p.add_argument("--stagnation-window", type=int, default=8)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-efficiency-gain", type=float, default=0.10)
    p.add_argument("--initial-skill")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in ("train", "validation", "test"):
        indices = value.get(key)
        if not isinstance(indices, list) or not indices or not all(isinstance(x, int) and x >= 0 for x in indices):
            raise ValueError(f"manifest {key!r} must be a non-empty list of non-negative integers")
        if len(indices) != len(set(indices)):
            raise ValueError(f"manifest {key!r} contains duplicate indices")
    return value


def round_batches(indices: list[int], rounds: int) -> list[list[int]]:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    batches = [[] for _ in range(min(rounds, len(indices)))]
    for position, index in enumerate(indices):
        batches[position % len(batches)].append(index)
    return batches


def relative_reduction(candidate: float, incumbent: float) -> float:
    if incumbent <= 0:
        return 0.0
    return (incumbent - candidate) / incumbent


def efficiency_aware_gate(
    candidate: dict[str, float],
    incumbent: dict[str, float],
    min_efficiency_gain: float,
) -> dict[str, Any]:
    """Protect reward first, then allow material efficiency improvements.

    Latency is reported but deliberately excluded from acceptance because it
    is hardware/load sensitive.  Steps, calls, and invalid actions are stable
    policy-level efficiency signals.
    """
    eps = 1e-9
    success_delta = candidate["success_rate"] - incumbent["success_rate"]
    score_delta = candidate["mean_score"] - incumbent["mean_score"]
    invalid_delta = candidate["invalid_actions"] - incumbent["invalid_actions"]
    no_reward_regression = success_delta >= -eps and score_delta >= -eps
    no_invalid_regression = invalid_delta <= eps
    primary_improvement = success_delta > eps or score_delta > eps
    gains = {
        "mean_steps": relative_reduction(candidate["mean_steps"], incumbent["mean_steps"]),
        "model_calls": relative_reduction(candidate["model_calls"], incumbent["model_calls"]),
        "invalid_actions": relative_reduction(candidate["invalid_actions"], incumbent["invalid_actions"]),
        "latency_seconds_diagnostic": relative_reduction(
            candidate["latency_seconds"], incumbent["latency_seconds"]
        ),
    }
    efficiency_improvement = (
        gains["mean_steps"] >= min_efficiency_gain
        or gains["model_calls"] >= min_efficiency_gain
        or gains["invalid_actions"] >= min_efficiency_gain
    )
    accepted = no_reward_regression and no_invalid_regression and (primary_improvement or efficiency_improvement)
    if not no_reward_regression:
        reason = "rejected: validation reward regressed"
    elif not no_invalid_regression:
        reason = "rejected: invalid actions increased"
    elif primary_improvement:
        reason = "accepted: validation reward improved without regression"
    elif efficiency_improvement:
        reason = "accepted: reward tied and policy efficiency improved materially"
    else:
        reason = "rejected: no material reward or policy-efficiency improvement"
    return {
        "accepted": accepted,
        "reason": reason,
        "criterion": {
            "reward_guard": "success_rate and mean_score must not regress",
            "invalid_action_guard": "total invalid actions must not increase",
            "secondary_metrics": ["mean_steps", "model_calls", "invalid_actions"],
            "minimum_relative_efficiency_gain": min_efficiency_gain,
            "latency": "diagnostic only; excluded from acceptance",
        },
        "deltas": {
            "success_rate": round(success_delta, 6),
            "mean_score": round(score_delta, 6),
            "invalid_actions": round(invalid_delta, 6),
        },
        "relative_efficiency_gains": {key: round(value, 6) for key, value in gains.items()},
    }


def write_version(directory: Path, version: int, skill: str, metadata: dict[str, Any]) -> None:
    versions = directory / "skills"
    versions.mkdir(parents=True, exist_ok=True)
    stem = f"skill_v{version:02d}"
    (versions / f"{stem}.md").write_text(skill, encoding="utf-8")
    save_json(versions / f"{stem}.json", metadata)


def dry_run(out: Path, manifest: dict[str, Any], args: argparse.Namespace, skill: str) -> int:
    tied_but_faster = efficiency_aware_gate(
        {"success_rate": 1.0, "mean_score": 1.0, "invalid_actions": 0.0, "mean_steps": 3.0,
         "model_calls": 3.0, "latency_seconds": 180.0},
        {"success_rate": 1.0, "mean_score": 1.0, "invalid_actions": 0.0, "mean_steps": 7.0,
         "model_calls": 7.0, "latency_seconds": 420.0},
        args.min_efficiency_gain,
    )
    reward_regression = efficiency_aware_gate(
        {"success_rate": 0.8, "mean_score": 0.8, "invalid_actions": 0.0, "mean_steps": 2.0,
         "model_calls": 2.0, "latency_seconds": 20.0},
        {"success_rate": 1.0, "mean_score": 1.0, "invalid_actions": 0.0, "mean_steps": 7.0,
         "model_calls": 7.0, "latency_seconds": 70.0},
        args.min_efficiency_gain,
    )
    assert tied_but_faster["accepted"]
    assert not reward_regression["accepted"]
    batches = round_batches(manifest["train"], args.rounds)
    write_version(out, 0, skill, {"version": 0, "sha": stable_sha(skill), "source": "initial"})
    result = {
        "ok": True,
        "manifest": manifest,
        "round_batches": batches,
        "planned_comparisons": ["test_no_skill", "test_static", "test_managed"],
        "gate_examples": {"tied_but_faster": tied_but_faster, "reward_regression": reward_regression},
    }
    save_json(out / "dry_run.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if not 0 <= args.min_efficiency_gain <= 1:
        raise SystemExit("--min-efficiency-gain must be between 0 and 1")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.split_manifest))
    static_skill = Path(args.initial_skill).read_text(encoding="utf-8") if args.initial_skill else DEFAULT_SKILL
    if args.dry_run:
        return dry_run(out, manifest, args, static_skill)

    runner = AlfWorldRunner(
        args.model, args.ollama_url, args.timeout, args.seed, args.action_retries, args.stagnation_window
    )
    skill = static_skill
    version = 0
    rejected_streak = 0
    history: list[dict[str, Any]] = []
    write_version(out, version, skill, {"version": version, "sha": stable_sha(skill), "source": "initial"})

    for round_number, train_batch in enumerate(round_batches(manifest["train"], args.rounds), start=1):
        prefix = f"round_{round_number:02d}"
        train_eps = run_set(runner, "train", train_batch, skill, args.max_steps, out, f"{prefix}_train")
        edit = propose_skill_edit(args.model, args.ollama_url, args.timeout, args.seed + round_number, skill, train_eps)
        candidate = edit["candidate_skill"]
        incumbent_eps = run_set(
            runner, "valid_seen", manifest["validation"], skill, args.max_steps, out, f"{prefix}_validation_incumbent"
        )
        candidate_eps = run_set(
            runner, "valid_seen", manifest["validation"], candidate, args.max_steps, out, f"{prefix}_validation_candidate"
        )
        incumbent_score = score(incumbent_eps)
        candidate_score = score(candidate_eps)
        gate = efficiency_aware_gate(candidate_score, incumbent_score, args.min_efficiency_gain)
        changed = stable_sha(candidate) != stable_sha(skill)
        gate["accepted"] = bool(gate["accepted"] and changed)
        if not changed:
            gate["reason"] = "rejected: candidate is identical to incumbent"
        record = {
            "round": round_number,
            "train_indices": train_batch,
            "incumbent_version": version,
            "incumbent_skill_sha": stable_sha(skill),
            "candidate_skill_sha": stable_sha(candidate),
            "incumbent": incumbent_score,
            "candidate": candidate_score,
            "gate": gate,
            "edit": edit,
        }
        if gate["accepted"]:
            version += 1
            skill = candidate
            rejected_streak = 0
            write_version(
                out, version, skill,
                {"version": version, "sha": stable_sha(skill), "parent_version": version - 1,
                 "accepted_in_round": round_number, "gate": gate, "edit_summary": edit.get("edit_summary", "")},
            )
            record["accepted_version"] = version
        else:
            rejected_streak += 1
        history.append(record)
        save_json(out / "rounds" / f"{prefix}.json", record)
        save_json(out / "update_history.json", history)
        if args.patience and rejected_streak >= args.patience:
            break

    no_skill_eps = run_set(
        runner, "valid_unseen", manifest["test"], "", args.max_steps, out, "test_no_skill"
    )
    static_eps = run_set(
        runner, "valid_unseen", manifest["test"], static_skill, args.max_steps, out, "test_static"
    )
    managed_eps = run_set(
        runner, "valid_unseen", manifest["test"], skill, args.max_steps, out, "test_managed"
    )
    (out / "best_skill.md").write_text(skill, encoding="utf-8")
    summary = {
        "model": args.model,
        "seed": args.seed,
        "manifest": manifest,
        "rounds_requested": args.rounds,
        "rounds_completed": len(history),
        "accepted_updates": sum(bool(item["gate"]["accepted"]) for item in history),
        "final_skill_version": version,
        "final_skill_sha": stable_sha(skill),
        "test_no_skill": score(no_skill_eps),
        "test_static": score(static_eps),
        "test_managed": score(managed_eps),
        "update_history": history,
    }
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
