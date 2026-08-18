from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.ingestion.models import Chunk, Document


class Chunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        file_type = document.metadata.get("file_type", "")

        if file_type == ".md":
            sections = self._markdown_splitter.split_text(document.content)

            chunks: list[Chunk] = []

            for section in sections:
                texts = self._text_splitter.split_text(section.page_content)

                for text in texts:
                    metadata = {
                        **document.metadata,
                        **{
                            key: str(value)
                            for key, value in section.metadata.items()
                        },
                    }

                    chunk_index = len(chunks)

                    chunks.append(
                        Chunk(
                            chunk_id=(
                                f"{document.document_id}:"
                                f"chunk-{chunk_index:04d}"
                            ),
                            document_id=document.document_id,
                            content=text,
                            metadata=metadata,
                        )
                    )
        else:
            texts = self._text_splitter.split_text(document.content)

            chunks = [
                Chunk(
                    chunk_id=f"{document.document_id}:chunk-{index:04d}",
                    document_id=document.document_id,
                    content=text,
                    metadata=dict(document.metadata),
                )
                for index, text in enumerate(texts)
            ]

        return chunks
