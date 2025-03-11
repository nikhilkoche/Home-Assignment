from vectordb import PineconeDB
import os
from langchain.schema import Document
import uuid
import fitz
import pdfplumber


class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.chunks = []
        self.tables_by_page = {}

    def extract_tables(self):
        """Extract tables from the PDF, ensuring no image-based tables are mistakenly captured."""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                table_texts = []

                for table in tables:
                    # Convert None values to empty strings
                    cleaned_table = [
                        " | ".join([cell if cell is not None else "" for cell in row])
                        for row in table
                    ]
                    table_text = "\n".join(cleaned_table)

                    # 🚨 **Ignore "Fake Tables" (short or numeric-only)**
                    if len(table_text.strip()) > 5 and not table_text.isnumeric():
                        table_texts.append(table_text)

                if table_texts:
                    self.tables_by_page[page_num] = table_texts

    def extract_text(self):
        """Extract text from the PDF, chunking properly into paragraphs."""
        for page_num, page in enumerate(self.doc):
            blocks = page.get_text("dict")["blocks"]
            prev_bottom = None
            current_chunk = []
            text_blocks = []

            for block in blocks:
                if block["type"] == 0:  # Text Block
                    for line in block["lines"]:
                        span_text = " ".join(span["text"] for span in line["spans"])

                        if not span_text.strip():
                            continue  # Ignore empty lines

                        top = line["bbox"][1]  # Get Y-position of text

                        # **Group text by spacing**
                        if prev_bottom is not None and (top - prev_bottom) > 15:
                            if len(" ".join(current_chunk)) > 100:
                                text_blocks.append(" ".join(current_chunk))
                            current_chunk = []

                        current_chunk.append(span_text)
                        prev_bottom = line["bbox"][3]

            # Add last chunk
            if current_chunk and len(" ".join(current_chunk)) > 100:
                text_blocks.append(" ".join(current_chunk))

            # Store extracted text along with associated tables
            for text in text_blocks:
                chunk_id = str(uuid.uuid4())
                chunk = {
                    "id": chunk_id,
                    "type": "text",
                    "text": text,
                    "metadata": {
                        "page": page_num + 1,
                        "associated_tables": self.tables_by_page.get(page_num, [])  # Attach extracted tables
                    }
                }
                self.chunks.append(chunk)
    from langchain.schema import Document

    def prepare_chunks_for_vectordb(self):
        """
        Convert extracted chunks into LangChain Document format.
        
        Args:
            chunks (list): Extracted text chunks from the PDF processing.
        
        Returns:
            List[Document]: Ready for vector database ingestion.
        """
        self.documents = []

        for chunk in self.chunks:
            text_content = chunk["text"]
            metadata = {
                "page": chunk["metadata"]["page"],
                "associated_tables": chunk["metadata"].get("associated_tables", [])  # Add tables if present
            }

            # Create a LangChain Document object
            document = Document(
                page_content=text_content,
                metadata=metadata
            )

            self.documents.append(document)

        return self.documents



    def process(self):
        """Run the full processing pipeline: extract tables and text."""
        self.extract_tables()
        self.extract_text()
        self.prepare_chunks_for_vectordb()
        
        return self.documents
