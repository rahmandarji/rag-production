import json
from pathlib import Path

import pytest

from app.evaluation.dataset import load_evaluation_dataset


def test_load_evaluation_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval.json"

    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "question": "What is this?",
                    "expected_answer": "A document.",
                    "expected_chunk_ids": ["chunk-1"],
                    "should_refuse": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_dataset(path)

    assert len(cases) == 1
    assert cases[0].case_id == "case-1"
    assert cases[0].expected_chunk_ids == ["chunk-1"]


def test_dataset_must_be_a_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval.json"

    path.write_text(
        json.dumps({"case_id": "invalid"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="JSON array",
    ):
        load_evaluation_dataset(path)


def test_missing_dataset(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_evaluation_dataset(
            tmp_path / "missing.json"
        )
