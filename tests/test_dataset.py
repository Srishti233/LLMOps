from src.data.dataset import generate_examples, read_jsonl, split_examples, write_jsonl


def test_generates_requested_number_of_unique_examples():
    rows = generate_examples(300, seed=42)
    assert len(rows) == 300
    assert len({r["instruction"] for r in rows}) == 300


def test_is_deterministic_for_a_seed_and_differs_across_seeds():
    assert generate_examples(50, 1) == generate_examples(50, 1)
    assert generate_examples(50, 1) != generate_examples(50, 2)


def test_every_keyword_is_contained_in_its_response():
    for row in generate_examples(300, 42):
        assert row["keywords"], row
        for kw in row["keywords"]:
            assert kw.lower() in row["response"].lower(), (kw, row["response"])


def test_all_topics_are_covered():
    from src.data.dataset import TOPICS

    assert {r["topic"] for r in generate_examples(300, 42)} == {t["name"] for t in TOPICS}


def test_split_has_no_overlap_and_keeps_everything():
    rows = generate_examples(100, 3)
    train, val = split_examples(rows, 0.2, 3)
    assert len(train) + len(val) == 100 and len(val) == 20
    assert not ({r["id"] for r in train} & {r["id"] for r in val})


def test_jsonl_roundtrip(tmp_path):
    rows = generate_examples(10, 5)
    path = tmp_path / "x" / "data.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert read_jsonl(tmp_path / "missing.jsonl") == []
