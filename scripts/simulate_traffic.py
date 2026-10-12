"""Send normal traffic, then deliberately shifted traffic, so drift and feedback can be demonstrated.

    python scripts/simulate_traffic.py --url http://localhost:8000 --normal 20 --shifted 10
    python scripts/simulate_traffic.py --direct-db --normal 40 --shifted 40   # no API/model needed
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import generate_examples  # noqa: E402
from src.monitor import store  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_utils import get_logger, log_fields, setup_logging  # noqa: E402

log = get_logger("simulate_traffic")

OFF_TOPIC = [
    "Explain in detail how to make a sourdough starter from scratch and keep it alive for several months while travelling.",
    "What is the best way to photograph the Orion nebula with a small backyard telescope and a DSLR camera?",
    "How did the Roman Empire organise taxation and grain supply across its distant provinces in the third century?",
    "Describe the offside rule and the most common pressing tactics used by modern football coaches in the league.",
    "Walk me through the main differences between a Roth retirement account and a traditional one for a freelancer.",
]


def normal_prompts(n: int, seed: int) -> list[str]:
    return [e["instruction"] for e in generate_examples(n, seed)]


def shifted_prompts(n: int, seed: int) -> list[str]:
    """Different topic and much longer prompts (repeated, lightly varied sentences)."""
    rng = random.Random(seed)
    return [" ".join(rng.choice(OFF_TOPIC) for _ in range(rng.randint(3, 7))) for _ in range(n)]


def http_json(method: str, url: str, payload: dict | None = None, timeout: int = 300) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as err:
        return err.code, {"detail": err.read().decode(errors="replace")[:200]}


def wait_ready(base: str, wait: int) -> None:
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            status, _ = http_json("GET", f"{base}/ready", timeout=5)
            if status == 200:
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    raise SystemExit(f"API at {base} not ready after {wait}s")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--normal", type=int, default=20)
    parser.add_argument("--shifted", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--feedback-rate", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait", type=int, default=300, help="seconds to wait for /ready")
    parser.add_argument("--direct-db", action="store_true", help="write fake logs straight into SQLite")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    plan = [("normal", p) for p in normal_prompts(args.normal, 10_000 + args.seed)] + \
           [("shifted", p) for p in shifted_prompts(args.shifted, args.seed)]
    db_path = resolve_path(load_config()["data"]["monitoring_db"])
    if args.direct_db:
        store.init_db(db_path)
    else:
        wait_ready(args.url, args.wait)

    sent = failed = fb = 0
    for kind, prompt in plan:
        # users are happy with in-domain answers and mostly unhappy with off-topic ones
        rating = ("up" if rng.random() < 0.9 else "down") if kind == "normal" else ("down" if rng.random() < 0.8 else "up")
        give_feedback = rng.random() < args.feedback_rate
        if args.direct_db:
            request_id = uuid.uuid4().hex
            store.log_request(db_path, request_id=request_id, prompt=prompt, response="[simulated response]",
                              latency_ms=rng.uniform(800, 4000), tokens_generated=rng.randint(10, 48),
                              model_version="simulated", prompt_version="v1")
            status = 200
            if give_feedback:
                store.log_feedback(db_path, request_id=request_id, rating=rating)
                fb += 1
        else:
            status, body = http_json("POST", f"{args.url}/generate",
                                     {"prompt": prompt, "max_new_tokens": args.max_new_tokens, "temperature": 0.2})
            if status == 200 and give_feedback:
                fstatus, _ = http_json("POST", f"{args.url}/feedback", {"request_id": body["request_id"], "rating": rating})
                fb += int(fstatus == 200)
        sent += 1
        failed += int(status != 200)
        log_fields(log, 20, "sent", kind=kind, status=status, chars=len(prompt))
    summary = {"sent": sent, "failed": failed, "feedback_sent": fb, "normal": args.normal, "shifted": args.shifted}
    print(json.dumps(summary))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
