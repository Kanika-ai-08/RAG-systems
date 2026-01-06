This repository will have all the basics impletation of RAG 

- Ingestion pipeline
      - Load all the documents
      - Chunking 
      - Create embeddings using openai embedding models (cosine algorithm)
      - Store vector embeddings to vector store (Chroma DB)

- Retrieval pipeline
     - Create vector embeddings of user query using same embedding model
     - Retrieve the context from the persisting vector database (cosine similarity algorithm).
  
  
