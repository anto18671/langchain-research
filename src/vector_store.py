from langchain_community.vectorstores import FAISS

# Create or load FAISS index
def build_vector_store(chunks, embedding_model):
    # Filter only valid text documents
    clean_chunks = []
    for doc in chunks:
        if isinstance(doc.page_content, str) and doc.page_content.strip():
            clean_chunks.append(doc)

    # Raise error if no chunks
    if not clean_chunks:
        raise ValueError("No valid text chunks found. Check your document loader.")

    # Create FAISS store
    store = FAISS.from_documents(clean_chunks, embedding_model)

    # Save index
    store.save_local("../data/vector_store")

    # Return store
    return store

# Load FAISS index
def load_vector_store(embedding_model):
    # Load store from directory
    store = FAISS.load_local(
        "./data/vector_store",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    # Return store
    return store
