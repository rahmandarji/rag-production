import json
from pathlib import Path

from app.evaluation.models import EvaluationCase


def load_evaluation_dataset(
    path: str | Path,
) -> list[EvaluationCase]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON array."
        )

    return [
        EvaluationCase.model_validate(item)
        for item in data
    ]
