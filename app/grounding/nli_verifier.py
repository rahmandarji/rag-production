import numpy as np
from sentence_transformers import CrossEncoder
from transformers import AutoConfig

from app.grounding.verifier import ClaimVerifier


class NLIVerifier(ClaimVerifier):
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        threshold: float = 0.8,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

        self.model_name = model_name
        self.threshold = threshold

        config = AutoConfig.from_pretrained(model_name)

        self.entailment_index = self._find_entailment_index(
            config.id2label
        )

        self.model = CrossEncoder(model_name)

    def verify(
        self,
        claim: str,
        evidence: list[str],
    ) -> list[tuple[str, float]]:
        if not claim.strip():
            raise ValueError("claim must not be empty.")

        if not evidence:
            return []

        pairs = [(text, claim) for text in evidence]

        logits = np.asarray(self.model.predict(pairs))

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        if logits.shape[1] != 3:
            raise ValueError(
                f"Expected 3 NLI logits, got shape {logits.shape}."
            )

        probabilities = self._softmax(logits)

        entailment_scores = probabilities[:, self.entailment_index]

        return [
            (text, float(score))
            for text, score in zip(evidence, entailment_scores)
            if float(score) >= self.threshold
        ]

    @staticmethod
    def _find_entailment_index(id2label: dict[int, str]) -> int:
        for index, label in id2label.items():
            if label.lower() == "entailment":
                return int(index)

        raise ValueError(
            f"Model configuration does not contain an entailment label: "
            f"{id2label}"
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(
            logits,
            axis=1,
            keepdims=True,
        )

        exponentials = np.exp(shifted)

        return exponentials / np.sum(
            exponentials,
            axis=1,
            keepdims=True,
        )
