from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

def create_vectorstore(data_dir: str, persist_dir: str) -> Chroma:
    """
    Create a Chroma vectorstore from PDF documents in a directory.

    Args:
        data_dir (str): Directory containing PDF files.
        persist_dir (str): Directory to persist the Chroma database.

    Returns:
        Chroma: The created Chroma vectorstore.
    """
    # Load PDF documents
    loader = PyPDFDirectoryLoader(data_dir)
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create and persist the vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )

    return vectorstore

if __name__ == "__main__":
    create_vectorstore("data/", "data/")