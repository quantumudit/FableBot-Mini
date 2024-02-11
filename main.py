"""
summary
"""

from src.pipelines.chain import Chain
from src.utils.basic_utils import wrap_text
from src.utils.llm_utils import get_llm_response

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
