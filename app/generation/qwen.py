from transformers import AutoModelForCausalLM, AutoTokenizer

from app.generation.models import AnswerSource, GeneratedAnswer
from app.generation.provider import GenerationProvider
from app.retrieval.models import RetrievalResult


REFUSAL_MESSAGE = (
    "I don't have enough information in the provided documents "
    "to answer this question."
)


class QwenGenerationProvider(GenerationProvider):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 256,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

    def generate(
        self,
        query: str,
        evidence: list[RetrievalResult],
    ) -> GeneratedAnswer:
        query = query.strip()

        if not query:
            raise ValueError("query must not be empty.")

        if not evidence:
            return GeneratedAnswer(
                answer=REFUSAL_MESSAGE,
                sources=[],
            )

        context_parts = []

        for index, result in enumerate(evidence, start=1):
            context_parts.append(
                f"[Evidence {index}]\n"
                f"{result.chunk.content}"
            )

        context = "\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document-grounded assistant. "
                    "You must answer using ONLY the supplied evidence. "
                    "Do not use outside knowledge. "
                    "Do not infer facts that are not supported by the evidence. "
                    f"If the evidence is insufficient, respond exactly: "
                    f'"{REFUSAL_MESSAGE}"'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{context}\n\n"
                    f"Question:\n{query}"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        generated_tokens = outputs[
            0
        ][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        if not answer:
            answer = REFUSAL_MESSAGE

        sources = [
            AnswerSource(
                chunk_id=result.chunk.chunk_id,
                document_id=result.chunk.document_id,
                content=result.chunk.content,
                metadata=result.chunk.metadata,
            )
            for result in evidence
        ]

        return GeneratedAnswer(
            answer=answer,
            sources=sources,
        )
