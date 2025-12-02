from loader import load_and_split_documents
from embedder import create_embedding_model
from vector_store import build_vector_store
from llm import load_local_llm
from rag_chain import build_rag_chain

# Run the KB system
def main():
    # Load documents
    chunks = load_and_split_documents("../data/docs")

    # Create embeddings
    embedding_model = create_embedding_model()

    # Build vector store
    store = build_vector_store(chunks, embedding_model)

    # Load local LLM
    tokenizer, model = load_local_llm()

    # Build RAG chain
    rag = build_rag_chain(store, tokenizer, model)

    # Start chat loop
    print("Knowledge Base Chat Ready. Type 'exit' to quit.")
    while True:
        # Get user input
        user_input = input("\nYou: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        # Get model response
        answer = rag(user_input)

        # Print response
        print("\nAssistant:", answer)

# Entry point
if __name__ == "__main__":
    main()
