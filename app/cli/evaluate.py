import json
from pathlib import Path

from app.core.container import create_rag_pipeline
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.runner import RAGEvaluator


def main() -> None:
    dataset_path = Path(
        "data/evaluation/rag_eval.json"
    )

    if not dataset_path.exists():
        raise SystemExit(
            f"Evaluation dataset does not exist: "
            f"{dataset_path}"
        )

    cases = load_evaluation_dataset(
        dataset_path
    )

    pipeline = create_rag_pipeline()

    evaluator = RAGEvaluator(
        pipeline=pipeline,
        retrieval_limit=5,
    )

    report = evaluator.evaluate(cases)

    print(
        json.dumps(
            report.model_dump(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
