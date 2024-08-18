import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embeddings
from langchain_openai import OpenAI

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a world-renowned expert in AWS and also an incredible teacher. Your goal is to teach me, a confused student.
Answer the question based off the following context:
{context}

---
Answer the question based on the context above: {question}
"""
def main():
    #Make the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type = str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    #Prepare the ChromaDB
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #Search the DB
    results = db.similarity_search_with_score(query_text, k=3)

    #Fill in prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query_text)
    
    model = OpenAI()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_responses = f"Response: {response_text}\nSources: {sources}"
    print(formatted_responses)
    return response_text

if __name__ == "__main__":
    main()

