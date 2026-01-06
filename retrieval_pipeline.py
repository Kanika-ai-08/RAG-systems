from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , SystemMessage

load_dotenv()

# Retrieve the existing vector store

persistence_directory = "db/chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistence_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw : space": "cosine"}
)

query = "in what year did tesla begin production of the roadster??"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.3}
# )

relevant_docs = retriever.invoke(query)
print(f"user query: {query}")

print("------Context-------")

for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}: ")
    print(f" Content preview: {doc.page_content}...")
    print("--" * 50)
    print("\n")

prompt_template = f"""You are an expert assistant that helps answer user queries based on the provided context. Use the context to provide accurate and concise answers.

Context:
{chr(10).join([doc.page_content for doc in relevant_docs])}

Question:
{query}
"""

model = ChatOpenAI(model="gpt-4o", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant that provides answers based on the given context."), 
    HumanMessage(content=prompt_template)
]

result = model.invoke(messages)

print("------Answer-------")
print(result.content)