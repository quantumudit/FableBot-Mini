"""
summary
"""

from os.path import exists, normpath

from src.components.openai_llm_embed import oepnai_embedding, openai_llm
from src.constants import CONFIGS, PARAMS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import read_yaml, wrap_text
from src.utils.llm_utils import (
    create_chromadb_vectorstore,
    create_text_chunks,
    get_chain,
    get_chromadb,
    get_llm_response,
    get_retriever,
)


class Chain:
    """summary"""

    def __init__(self):
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).data_configs
        self.params = read_yaml(PARAMS)

        # Input docs
        self.docs = normpath(self.configs.data_directory)
        self.is_dir = self.configs.is_dir
        self.filetype = self.configs.filetype

        # Output db
        self.db = normpath(self.configs.vectorstore_directory)
        self.create_db = self.configs.create_db

        # Get text splitter parameters
        self.chunk_size = self.params.text_splitter_params.chunk_size
        self.chunk_ol = self.params.text_splitter_params.chunk_overlap

        # Get retriever parameters
        self.k = self.params.retriever_params.k

        # Get retriever chain parameters
        self.chain_type = self.params.chain_params.chain_type
        self.return_source = self.params.chain_params.return_source

    def create_chain(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        try:
            if not exists(self.db) or self.create_db:
                # Load and process text
                logger.info("Creating the text chunks from raw documents")
                text_chunks = create_text_chunks(
                    path=self.docs,
                    file_type=self.filetype,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_ol,
                    is_directory=self.is_dir,
                )
                logger.info("Text chunks created successfully")

                # Creating the vector store
                logger.info("Creating the vector store to save embeddings")
                vector_db = create_chromadb_vectorstore(
                    self.db, documents=text_chunks, embeddings=oepnai_embedding
                )
                logger.info(
                    "Vectorstore created at: %s and embeddings saved successfully",
                    self.db,
                )
            else:
                logger.info("Vectorstore exists. Loading the vector database")
                vector_db = get_chromadb(self.db, embedding_fx=oepnai_embedding)
                logger.info("Vectorstore loaded successfully")

            # Create retriever
            logger.info("Creating the retriever object")
            retriever = get_retriever(vector_db, retriever_k=self.k)
            logger.info("Retriever object created successfully")
            logger.info("The retriever search type is: %s", retriever.search_type)
            logger.info(
                "The retriever search arguments are: %s", retriever.search_kwargs
            )

            # Get Retrieval chain
            logger.info("Creating the retrieval Q&A chain")
            qa_chain_obj = get_chain(
                llm=openai_llm,
                retriever=retriever,
                chain_type=self.chain_type,
                return_src_docs=self.return_source,
            )
            logger.info("The retrieval Q&A chain created successfully")
            logger.info("Chatbot is ready to answer questions...")
            return qa_chain_obj
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e


if __name__ == "__main__":
    # create the Q&A chain
    chain_obj = Chain()
    qa_chain = chain_obj.create_chain()
    print("\n\n")

    # Whether to return source documents or, not
    while True:
        # Take question from user user
        query = input("Question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        return_source = input("View Sources (True/False): ")
        # Extract LLM response
        response = get_llm_response(query, chain=qa_chain, return_source_docs=True)

        # Extract answers & sources from LLM response
        ANSWER = wrap_text(response.get("response"))
        sources = response.get("source_docs")

        # Print the response and sources
        if return_source.lower() == "true":
            print("\n\nAnswer:")
            print(ANSWER)
            print("\nSources:")
            print("\n".join(sources))
            print("\n")
        else:
            print("\n\nAnswer:")
            print(ANSWER)
            print("\n")
