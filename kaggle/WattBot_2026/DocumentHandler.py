"""WattBot Markdown -> section-aware LangChain documents."""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class DocumentHandler:
    """
    Convert preprocessed WattBot Markdown files into
    section-aware LangChain Documents.

    Metadata:
        metadata.csv -> all columns are preserved as metadata.

    Markdown:
        processed_markdown/{document_id}/document.md

    Processing:
        MarkdownHeaderTextSplitter -> sections
        RecursiveCharacterTextSplitter -> large-section chunks
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        markdown_root: str | Path = "processed_markdown",
        max_chars: int = 4000,
        chunk_overlap: int = 400,
    ):
        self.metadata_csv = Path(metadata_csv)
        self.markdown_root = Path(markdown_root)

        self.max_chars = max_chars
        self.chunk_overlap = chunk_overlap

        self.paper_metadata = self._load_metadata()

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

    # =========================================================
    # Metadata
    # =========================================================

    def _load_metadata(self) -> dict[str, dict]:
        """
        Load metadata.csv using pandas.

        Every column present in metadata.csv is preserved.
        Missing values are converted to empty strings.
        """

        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {self.metadata_csv}"
            )

        df = pd.read_csv(self.metadata_csv, dtype=str, keep_default_na=False)

        df.columns = [str(column).strip() for column in df.columns]

        if "id" not in df.columns:
            raise ValueError("metadata.csv must contain an 'id' column.")

        metadata = {}

        for _, row in df.iterrows():
            document_id = str(row["id"]).strip()

            if not document_id:
                continue

            row_metadata = {}

            for column, value in row.items():
                if pd.isna(value):
                    value = ""

                value = str(value).strip()

                if value.lower() in {"nan", "null", "none"}:
                    value = ""

                row_metadata[column] = value

            metadata[document_id] = row_metadata

        return metadata

    # =========================================================
    # Markdown helpers
    # =========================================================

    @staticmethod
    def _extract_headings(section: Document) -> list[str]:
        headings = []

        for key, value in sorted(
            section.metadata.items(),
            key=lambda item: (
                int(item[0][1:])
                if item[0].startswith("h") and item[0][1:].isdigit()
                else 999
            ),
        ):
            if key.startswith("h") and key[1:].isdigit() and value:
                heading = str(value).strip()

                if heading:
                    headings.append(heading)

        return headings

    @staticmethod
    def _remove_empty_lines(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    # =========================================================
    # Markdown loading
    # =========================================================

    def _get_markdown_path(
        self,
        document_id: str,
    ) -> Path:
        """
        Resolve the already-processed Markdown file.

        Expected structure:

            processed_markdown/
                {document_id}/
                    document.md
        """

        return self.markdown_root / document_id / "document.md"

    def _load_markdown(
        self,
        document_id: str,
    ) -> str:
        markdown_path = self._get_markdown_path(document_id)

        if not markdown_path.exists():
            raise FileNotFoundError(
                f"Markdown file not found: {markdown_path}"
            )

        if not markdown_path.is_file():
            raise FileNotFoundError(
                f"Markdown path is not a file: {markdown_path}"
            )

        markdown = markdown_path.read_text(encoding="utf-8")

        return markdown or ""

    # =========================================================
    # Main processing
    # =========================================================

    def get_chunks_from_document(
        self,
        document_id: str,
    ) -> list[Document]:
        """
        Prepare a document from an already-generated Markdown file.

        The input path is used only to determine the document ID
        from its stem.

        Example:
            path = papers/amazon2023.pdf

            Markdown used:
                processed_markdown/amazon2023/document.md
        """

        print(f"rag.document.loading_started | document_id={document_id}")

        # -----------------------------------------------------
        # Load metadata
        # -----------------------------------------------------

        metadata = self.paper_metadata.get(document_id)

        if metadata is None:
            raise KeyError(
                f"Document ID '{document_id}' not found in metadata CSV."
            )

        # -----------------------------------------------------
        # Load existing Markdown
        # -----------------------------------------------------

        markdown = self._load_markdown(document_id)

        if not markdown.strip():
            return []

        markdown = self._remove_empty_lines(markdown).strip()

        # -----------------------------------------------------
        # Markdown -> sections
        # -----------------------------------------------------

        sections = self.markdown_splitter.split_text(markdown)

        chunks: list[Document] = []

        # Metadata columns whose values should be added to
        # the semantic prefix.
        prefix_columns = [
            "type",
            "title",
            "year",
            "citation",
            "url",
            "peer_reviewed",
            "venue_evidence",
            "venue",
        ]

        for section_index, section in enumerate(sections):
            section_headings = self._extract_headings(section)

            # -------------------------------------------------
            # Semantic prefix
            # -------------------------------------------------

            prefix_parts = []

            for column in prefix_columns:
                value = metadata.get(column, "").strip()

                if value:
                    prefix_parts.append(f"{column}-{value}")

            prefix_parts.extend(section_headings)

            heading_prefix = "\n".join(prefix_parts)

            if heading_prefix:
                heading_prefix += "\n\n"

            # -------------------------------------------------
            # Split large sections
            # -------------------------------------------------

            sub_chunks = self.recursive_splitter.split_text(
                section.page_content
            )

            # print(section.page_content)
            # print("\n\n\n\n")

            for chunk_index, chunk in enumerate(sub_chunks):
                chunk = chunk.strip()

                if not chunk:
                    continue

                chunk_content = heading_prefix + chunk

                # Preserve every field from metadata.csv.
                chunk_metadata = dict(metadata)

                # Add only chunk/document processing metadata.
                chunk_metadata.update(
                    {
                        "section_headings": ", ".join(section_headings),
                    }
                )

                chunks.append(
                    Document(
                        page_content=chunk_content,
                        metadata=chunk_metadata,
                    )
                )

        print(
            f"rag.document.processing_completed | "
            f"document_id={document_id} | "
            f"chunks={len(chunks)}"
        )

        return chunks