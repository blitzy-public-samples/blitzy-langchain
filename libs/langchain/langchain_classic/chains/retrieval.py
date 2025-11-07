from __future__ import annotations

from typing import Any

from langchain_core.retrievers import (
    BaseRetriever,
    RetrieverOutput,
)
from langchain_core.runnables import Runnable, RunnablePassthrough


def create_retrieval_chain(
    retriever: BaseRetriever | Runnable[dict, RetrieverOutput],
    combine_docs_chain: Runnable[dict[str, Any], str],
) -> Runnable:
    """Create retrieval chain that retrieves documents and then passes them on.

    This function composes a retrieval-augmented generation (RAG) chain that:
    1. Retrieves relevant documents based on user query
    2. Passes retrieved documents along with original inputs to a combine chain
    3. Returns both the retrieved context and the generated answer

    **Type Flow Documentation:**
    
    Input Dict ({"input": str, ...}) 
    → Retrieval Step → Documents (List[Document])
    → Combine Step (context + original inputs) → Answer (str)
    → Output Dict ({"input": str, "context": List[Document], "answer": str, ...})

    At each pipe stage:
    - Stage 1: Input dict with "input" key → retriever.invoke() → List[Document]
    - Stage 2: Original dict merged with {"context": List[Document]} → 
               combine_docs_chain.invoke() → str (answer)
    - Stage 3: Result merged into dict → {"context": [...], "answer": "...", ...}

    Args:
        retriever: Retriever-like object that returns list of documents.
            
            **Type Details:**
            - Accepts: BaseRetriever OR Runnable[dict, RetrieverOutput]
            - RetrieverOutput is defined as: List[Document]
            
            **Behavior by Type:**
            - If BaseRetriever: Automatically extracts "input" key from incoming
              dict and passes it to retriever. The retriever's invoke() method
              expects a string query, so {"input": "query"} → "query" extraction
              happens automatically via lambda function.
            - If generic Runnable[dict, RetrieverOutput]: Receives the entire 
              input dict. Your custom runnable must handle dict structure 
              internally and return List[Document].
            
            **Input Key Requirements:**
            - For BaseRetriever: Input dict MUST contain "input" key with string
              query value. Example: {"input": "What is LangChain?"}
            - For custom Runnable: No required keys, your implementation defines
              the expected dict structure.

        combine_docs_chain: Runnable that combines retrieved documents with inputs
            to produce final answer string.
            
            **Type Details:**
            - Signature: Runnable[dict[str, Any], str]
            - Input dict structure:
              * "context": List[Document] - The retrieved documents
              * "input": str - The original user query (passed through)
              * "chat_history": List - Conversation history (defaults to [] if 
                not in original inputs, enabling conversational retrieval)
              * Any additional keys from the original input dict are passed through
            - Output: str - The generated answer/response
            
            **Typical Implementation:**
            Created via create_stuff_documents_chain() which formats documents
            into a prompt and sends to an LLM for answer generation.

    Returns:
        Runnable[dict[str, Any], dict[str, Any]]: An LCEL Runnable chain.
        
        **Input Schema:**
        - Must be a dictionary
        - Required keys depend on retriever type (see Args section)
        - Typical input: {"input": "user query string"}
        
        **Output Schema:**
        - Type: dict[str, Any]
        - Guaranteed keys:
          * "context": List[Document] - All retrieved documents with metadata
          * "answer": str - The final generated answer from combine_docs_chain
        - Pass-through keys:
          * All keys from original input dict are preserved in output
          * Example: If input was {"input": "query", "user_id": 123}, output
            includes "user_id": 123
        
        **Example Output Structure:**
        ```python
        {
            "input": "What is LangChain?",
            "context": [
                Document(page_content="LangChain is a framework...", metadata={...}),
                Document(page_content="It enables...", metadata={...})
            ],
            "answer": "LangChain is a framework for developing applications..."
        }
        ```

    Raises:
        KeyError: If input dict is missing required "input" key when using 
            BaseRetriever. The lambda function `lambda x: x["input"]` will raise
            KeyError if "input" key is not present.
            
            Example trigger: 
            ```python
            retrieval_chain.invoke({"query": "..."})  # Wrong key name
            # Raises: KeyError: 'input'
            ```
        
        ValidationError: If combine_docs_chain has Pydantic input validation and
            the merged dict (original inputs + context) fails validation.
            
            Example trigger: combine_docs_chain expects specific field types but
            receives incompatible types from input dict or context.
        
        TypeError: If retriever does not return List[Document] or if 
            combine_docs_chain does not return str. Type mismatches in LCEL
            composition will raise TypeError during invoke().
        
        Any exceptions from retriever execution: Network errors, API timeouts,
            authentication failures, or retriever-specific errors are propagated.
        
        Any exceptions from combine_docs_chain execution: LLM API errors, rate
            limits, prompt template errors, or parsing failures are propagated.

    Example:
        **Complete Example with Type Flow Comments:**
        
        ```python
        # Installation: pip install -U langchain langchain-community langchain-openai
        import os
        from langchain_community.vectorstores import Chroma
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_classic.chains.combine_documents import (
            create_stuff_documents_chain,
        )
        from langchain_classic.chains import create_retrieval_chain

        # Setup: Configure API key
        os.environ["OPENAI_API_KEY"] = "your-api-key-here"

        # Step 1: Create a retriever (BaseRetriever subclass)
        # This example uses Chroma vector store as the retriever
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=[
                Document(page_content="LangChain is a framework for LLM apps"),
                Document(page_content="It provides chains, agents, and memory"),
            ],
            embedding=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        # Type: BaseRetriever (will extract "input" key automatically)

        # Step 2: Create combine_docs_chain
        # This chain takes documents and produces an answer
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using the following context:\\n\\n{context}"),
            ("human", "{input}"),
        ])
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        # Type: Runnable[dict[str, Any], str]
        # Input: {"context": List[Document], "input": str, ...}
        # Output: str (the answer)

        # Step 3: Create the retrieval chain
        # Type flow: dict → List[Document] → str → dict
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # Step 4: Invoke with sample input
        # Input type: dict with "input" key
        result = retrieval_chain.invoke({"input": "What is LangChain?"})
        
        # Expected output structure:
        # {
        #     "input": "What is LangChain?",
        #     "context": [
        #         Document(page_content="LangChain is a framework...", ...),
        #         Document(page_content="It provides chains...", ...)
        #     ],
        #     "answer": "LangChain is a framework for building applications..."
        # }
        print(f"Answer: {result['answer']}")
        print(f"Number of source documents: {len(result['context'])}")
        ```

    Source: libs/langchain/langchain_classic/chains/retrieval.py:12-68
    """
    # Type Flow Path Selection:
    # Determine how to invoke the retriever based on its type
    if not isinstance(retriever, BaseRetriever):
        # Path 1: Generic Runnable[dict, RetrieverOutput]
        # The retriever already accepts dict input, use directly
        # Type: Runnable[dict, List[Document]]
        retrieval_docs: Runnable[dict, RetrieverOutput] = retriever
    else:
        # Path 2: BaseRetriever (expects string input)
        # Extract "input" key from dict and pass to retriever
        # Type flow: dict → lambda extracts x["input"] → str → BaseRetriever → List[Document]
        # This creates: Runnable[dict, str] | Runnable[str, List[Document]] → Runnable[dict, List[Document]]
        retrieval_docs = (lambda x: x["input"]) | retriever

    # LCEL Composition - Type Flow Through Chain:
    # 
    # Stage 1: RunnablePassthrough.assign(context=retrieval_docs)
    #   - Takes input dict, passes it through unchanged
    #   - Invokes retrieval_docs with input dict → gets List[Document]
    #   - Merges result: input_dict.update({"context": List[Document]})
    #   - Type: Runnable[dict, dict] where output dict includes "context" key
    #   - with_config(run_name="retrieve_documents") adds tracing/observability name
    #
    # Stage 2: .assign(answer=combine_docs_chain)
    #   - Takes dict from Stage 1 (includes "context" and all original keys)
    #   - Invokes combine_docs_chain with merged dict → gets str (answer)
    #   - Merges result: dict.update({"answer": str})
    #   - Type: Runnable[dict, dict] where output dict includes "context" and "answer"
    #
    # Final: .with_config(run_name="retrieval_chain")
    #   - Wraps entire chain with tracing/observability name for monitoring
    #   - Does not change type or behavior, only adds metadata for callbacks/tracers
    #
    # Overall Type: Runnable[dict[str, Any], dict[str, Any]]
    # Input: {"input": str, ...} → Output: {"input": str, "context": List[Document], "answer": str, ...}
    return (
        RunnablePassthrough.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents"),
        ).assign(answer=combine_docs_chain)
    ).with_config(run_name="retrieval_chain")
