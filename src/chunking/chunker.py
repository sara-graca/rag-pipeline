from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path
import re

def create_chunks():
    markdown_document = Path("data/parsed/rulebook_final.md").read_text(encoding="utf-8")

    markdown_document = re.sub(r"<!-- page \d+ -->\n?", "", markdown_document) # strip the page markers because yhey only add noise

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
        
    ]

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50, separators=["\\n\\n", "\\n", " ", "", "- "] )

    final_chunks = text_splitter.split_documents(md_header_splits)

    return final_chunks

if __name__ == "__main__":
    main()
