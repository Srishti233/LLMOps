from types import SimpleNamespace

from src.evaluate import evaluate_samples
from src.utils.text_metrics import keyword_score, rouge_l


def test_rouge_l_bounds_and_values():
    assert rouge_l("a b c", "a b c") == 1.0
    assert rouge_l("x y", "a b") == 0.0
    assert rouge_l("", "a") == 0.0
    # LCS("a b c d", "a c d e") = 3 -> P=3/4 R=3/4
    assert abs(rouge_l("a b c d", "a c d e") - 0.75) < 1e-9


def test_keyword_score():
    assert keyword_score("Go to Settings  >  Billing", ["settings > billing"]) == (1.0, True)
    assert keyword_score("nothing", ["a", "b"]) == (0.0, False)
    cov, exact = keyword_score("has a only", ["a", "b"])
    assert cov == 0.5 and exact is False
    assert keyword_score("x", []) == (1.0, True)


def test_evaluate_samples_with_fake_generator():
    samples = [
        {"id": "1", "instruction": "q1", "response": "go to Settings now", "keywords": ["Settings"]},
        {"id": "2", "instruction": "q2", "response": "wait 30 days", "keywords": ["30 days"]},
    ]
    answers = {"q1": "go to Settings now", "q2": "unrelated"}

    def fake(question, max_new_tokens):
        return SimpleNamespace(text=answers[question], tokens=10, gen_seconds=2.0)

    result = evaluate_samples(samples, fake, 16)
    m = result["metrics"]
    assert result["n_samples"] == 2
    assert m["keyword_match_rate"] == 0.5 and m["keyword_coverage"] == 0.5
    assert abs(m["rouge_l"] - 0.5) < 1e-9
    assert m["avg_latency_s"] == 2.0 and m["tokens_per_sec"] == 5.0
