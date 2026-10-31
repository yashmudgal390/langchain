from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 1. Load the PDF
pdf_path = r"D:\langchain\text_splitter\dl-curriculum.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print("PDF loaded successfully!")
print("Total pages:", len(docs))

# 2. Initialize the text splitter
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator=" "
)

# 3. Split documents into chunks
chunks = splitter.split_documents(docs)

print("\nTotal chunks:", len(chunks))

# 4. Display chunks
for i, chunk in enumerate(chunks):
    print(f"\n{'=' * 50}")
    print(f"CHUNK {i + 1}")
    print(f"Page: {chunk.metadata.get('page', 'Unknown')}")
    print(f"{'=' * 50}")

    print(chunk.page_content)