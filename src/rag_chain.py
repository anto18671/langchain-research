from transformers import pipeline

# Build a retrieval-augmented answering function
def build_rag_chain(store, tokenizer, model):
    # Create retriever
    retriever = store.as_retriever(search_kwargs={"k": 4})

    # Create LLM pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )

    # Define chain logic
    def answer_question(query):
        # Retrieve context
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        # Build prompt
        prompt = (
            "Use the context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        # Generate answer
        response = generator(prompt)[0]["generated_text"]

        # Return answer
        return response

    # Return chain function
    return answer_question
