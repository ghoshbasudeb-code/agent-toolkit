from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_index(data_dir: str = "data", save_path: str = "vectorstore"):
    """Loads text files from a directory, chunks them, and generates a FAISS vector index."""
    # 1. Load documents
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print(f"No documents found in '{data_dir}'. Please add .txt files to ingest.")
        return

    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 3. Generate Embeddings & Save to Vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"Vector store created and saved to '{save_path}'")


if __name__ == "__main__":
    build_index()
