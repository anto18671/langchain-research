from langchain_community.embeddings import HuggingFaceEmbeddings

# Create embedding model
def create_embedding_model():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Return embeddings
    return embeddings
