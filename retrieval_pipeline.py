from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

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
