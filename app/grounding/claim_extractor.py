import re


class ClaimExtractor:
    """Extract factual claims from a generated answer."""

    def extract(self, answer: str) -> list[str]:
        if not answer.strip():
            return []

        claims = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
            if sentence.strip()
        ]

        return claims
