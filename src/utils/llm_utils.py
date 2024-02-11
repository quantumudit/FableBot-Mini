"""
summary
"""

from os import makedirs
from os.path import normpath

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma


def create_text_chunks(
    path: str,
    file_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    is_directory: bool = True,
):
    """_summary_

    Args:
        path (str): _description_
        file_type (str): _description_
        chunk_size (int, optional): _description_. Defaults to 1000.
        chunk_overlap (int, optional): _description_. Defaults to 200.
        is_directory (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if is_directory:
        dir_path = normpath(path)
        if file_type == "text":
            loader = DirectoryLoader(dir_path, glob="./*.txt", loader_cls=TextLoader)
        if file_type == "pdf":
            loader = DirectoryLoader(dir_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    else:
        file_path = normpath(path)
        if file_type == "text":
            loader = TextLoader(file_path)
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)

    # Load the documents
    documents = loader.load()

    # Create splitter object
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Apply splitter object on the documents
    texts = text_splitter.split_documents(documents)

    return texts


def create_chromadb_vectorstore(db_dirpath: str, documents, embeddings):
    """_summary_

    Args:
        db_dirpath (str): _description_
        documents (_type_): _description_
        embeddings (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Normalize the directory path
    db_path = normpath(db_dirpath)

    # Create the database directory (if not exists)
    makedirs(db_path, exist_ok=True)

    # Create the vector database and save the embeddings in the vector database
    vector_db = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=db_dirpath
    )
    return vector_db


def get_chromadb(vectordb_path: str, embedding_fx):
    """_summary_

    Args:
        vectordb_path (str): _description_
        embedding_fx (_type_): _description_
        as_retriever (bool, optional): _description_. Defaults to True.
        retriever_k (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    # Normalize the vector database path
    db_path = normpath(vectordb_path)

    # Load the vector database
    vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_fx)
    return vector_db


def get_retriever(vector_db, retriever_k: int = 3):
    """_summary_

    Args:
        vector_db (_type_): _description_
        retriever_k (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    retriever = vector_db.as_retriever(search_kwargs={"k": retriever_k})
    return retriever


def get_chain(llm, retriever, chain_type: str = "stuff", return_src_docs: bool = True):
    """_summary_

    Args:
        llm (_type_): _description_
        retriever (_type_): _description_
        chain_type (str, optional): _description_. Defaults to "stuff".
        return_src_docs (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=return_src_docs,
    )
    return qa_chain


def get_llm_response(query, chain, return_source_docs: bool = True):
    """_summary_

    Args:
        query (_type_): _description_
        chain (_type_): _description_
        return_source_docs (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    response = chain.invoke(query)
    if return_source_docs:
        source_docs = [src.metadata["source"] for src in response["source_documents"]]
        return {"response": response["result"], "source_docs": source_docs}

    return response["result"]
