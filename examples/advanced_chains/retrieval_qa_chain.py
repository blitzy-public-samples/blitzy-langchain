#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) Example with LangChain.

This example demonstrates how to implement a complete RAG pipeline using
LangChain's create_retrieval_chain with an in-memory vector store. RAG is a
pattern that enhances LLM responses by retrieving relevant context from a
knowledge base before generating answers.

This example follows the Agent Action Plan section 0.5.1 requirements for
executable examples with complete imports, sample data, error handling, and
detailed inline documentation.

Source: examples/advanced_chains/retrieval_qa_chain.py
Related: libs/langchain/tests/unit_tests/chains/test_retrieval.py
Related: libs/core/tests/unit_tests/vectorstores/test_in_memory.py
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Core LangChain imports for RAG implementation
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

# LangChain classic chains (create_retrieval_chain factory)
from langchain_classic.chains import create_retrieval_chain

# Try to import OpenAI components (gracefully handle missing API key)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: langchain-openai not installed. Install with: "
          "pip install langchain-openai")

# Fallback fake embeddings for testing without API keys
try:
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding
    FAKE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    FAKE_EMBEDDINGS_AVAILABLE = False


def create_sample_documents() -> List[Document]:
    """
    Create sample documents with realistic LangChain concept content.
    
    In a production system, these documents would come from:
    - Document loaders (PDFs, websites, databases)
    - Text splitters for chunking large documents
    - Existing knowledge bases
    
    Returns:
        List[Document]: List of Document objects with page_content and metadata.
    
    Example:
        >>> docs = create_sample_documents()
        >>> len(docs)
        5
        >>> docs[0].page_content
        'LangChain is a framework for developing applications...'
    """
    # Sample documents covering key LangChain concepts
    documents = [
        Document(
            page_content=(
                "LangChain is a framework for developing applications powered "
                "by language models. It enables applications that are "
                "context-aware and can reason. The framework provides modular "
                "components including chains, agents, and retrievers that can "
                "be composed together."
            ),
            metadata={
                "source": "langchain_overview",
                "topic": "introduction",
                "relevance": "high"
            }
        ),
        Document(
            page_content=(
                "LCEL (LangChain Expression Language) is a declarative way to "
                "compose chains. It uses the pipe operator (|) to connect "
                "components. For example: prompt | llm | parser creates a "
                "chain that processes input through a prompt template, sends "
                "it to an LLM, and parses the output. LCEL chains support "
                "streaming, async, and batching out of the box."
            ),
            metadata={
                "source": "lcel_documentation",
                "topic": "composition",
                "relevance": "high"
            }
        ),
        Document(
            page_content=(
                "Retrieval-Augmented Generation (RAG) is a pattern that "
                "retrieves relevant documents from a knowledge base before "
                "generating a response. The create_retrieval_chain function "
                "builds a chain that: 1) retrieves relevant documents based "
                "on the query, 2) combines the documents with the original "
                "query, and 3) generates an answer using the LLM with full "
                "context."
            ),
            metadata={
                "source": "rag_guide",
                "topic": "retrieval",
                "relevance": "high"
            }
        ),
        Document(
            page_content=(
                "Vector stores enable semantic search by storing document "
                "embeddings. InMemoryVectorStore is a simple implementation "
                "that stores vectors in memory. It supports similarity search, "
                "MMR (Maximum Marginal Relevance), and can be easily swapped "
                "with production vector databases like Chroma, Pinecone, or "
                "Weaviate."
            ),
            metadata={
                "source": "vectorstore_guide",
                "topic": "storage",
                "relevance": "medium"
            }
        ),
        Document(
            page_content=(
                "Embeddings convert text into dense vector representations "
                "that capture semantic meaning. OpenAI's text-embedding-ada-002 "
                "model creates 1536-dimensional embeddings. Similar documents "
                "have vectors close together in the embedding space, enabling "
                "semantic search beyond keyword matching."
            ),
            metadata={
                "source": "embeddings_guide",
                "topic": "embeddings",
                "relevance": "medium"
            }
        ),
    ]
    
    return documents


def get_embeddings_model() -> Optional[Embeddings]:
    """
    Initialize embeddings model with graceful fallback handling.
    
    Tries to use OpenAI embeddings if API key is available, otherwise
    falls back to deterministic fake embeddings for testing/demo purposes.
    
    Returns:
        Optional[Embeddings]: Embeddings model instance, or None if unavailable.
    
    Raises:
        ValueError: If neither OpenAI nor fake embeddings are available.
    
    Example:
        >>> embeddings = get_embeddings_model()
        >>> # Will use OpenAI if OPENAI_API_KEY is set, else fake embeddings
    """
    # Check for OpenAI API key in environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key and OPENAI_AVAILABLE:
        print("✓ Using OpenAI embeddings (text-embedding-ada-002)")
        try:
            return OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=openai_api_key
            )
        except Exception as e:
            print(f"⚠ Failed to initialize OpenAI embeddings: {e}")
            print("  Falling back to fake embeddings...")
    else:
        if not openai_api_key:
            print("⚠ OPENAI_API_KEY not found in environment")
        if not OPENAI_AVAILABLE:
            print("⚠ langchain-openai not installed")
        print("  Using fake embeddings for demonstration")
    
    # Fallback to fake embeddings
    if FAKE_EMBEDDINGS_AVAILABLE:
        # DeterministicFakeEmbedding creates consistent embeddings for testing
        # In production, always use real embeddings for semantic accuracy
        return DeterministicFakeEmbedding(size=1536)
    else:
        raise ValueError(
            "No embeddings available. Install langchain-openai and set "
            "OPENAI_API_KEY, or install langchain-core for fake embeddings."
        )


def get_chat_model() -> Optional[Runnable]:
    """
    Initialize chat model with graceful error handling.
    
    Returns:
        Optional[Runnable]: Chat model instance, or None if unavailable.
    
    Raises:
        ValueError: If chat model cannot be initialized.
    
    Example:
        >>> llm = get_chat_model()
        >>> # Returns ChatOpenAI if API key available
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in your .env file or environment:\n"
            "export OPENAI_API_KEY='your-api-key-here'"
        )
    
    if not OPENAI_AVAILABLE:
        raise ValueError(
            "langchain-openai package is required. Install with:\n"
            "pip install langchain-openai"
        )
    
    print("✓ Using ChatOpenAI (gpt-3.5-turbo)")
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,  # Deterministic responses for QA
        openai_api_key=openai_api_key
    )


def create_combine_docs_chain(llm: Runnable) -> Runnable:
    """
    Create a chain for combining retrieved documents with the query.
    
    This chain formats the retrieved context documents and generates an
    answer based on the provided context. It demonstrates LCEL composition
    with prompt | llm | parser pattern.
    
    Args:
        llm: Language model for generating answers.
    
    Returns:
        Runnable: Chain that takes context and input, returns answer string.
    
    Example:
        >>> llm = get_chat_model()
        >>> chain = create_combine_docs_chain(llm)
        >>> # Chain expects {"context": [...], "input": "question"}
    """
    # Define prompt template for QA with context
    # This template instructs the LLM to answer based only on provided context
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context.
If you cannot answer the question based on the context, say "I don't have "
enough information to answer that question."

Context:
{context}

Question: {input}

Answer:"""
    )
    
    # LCEL composition: prompt | llm | parser
    # Type flow: Dict[str, Any] → List[BaseMessage] → AIMessage → str
    combine_docs_chain = prompt | llm | StrOutputParser()
    
    return combine_docs_chain


def format_docs(docs: List[Document]) -> str:
    """
    Format list of documents into a single string for context.
    
    Args:
        docs: List of retrieved Document objects.
    
    Returns:
        str: Formatted string with all document contents separated by newlines.
    
    Example:
        >>> docs = [Document(page_content="Doc 1"), Document(page_content="Doc 2")]
        >>> formatted = format_docs(docs)
        >>> print(formatted)
        Doc 1

        Doc 2
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    vectorstore: InMemoryVectorStore,
    llm: Runnable,
    search_kwargs: Optional[Dict[str, Any]] = None
) -> Runnable:
    """
    Build complete RAG chain using create_retrieval_chain.
    
    This demonstrates the full RAG workflow:
    1. Query comes in as {"input": "question"}
    2. Retriever searches vector store for relevant documents
    3. Documents are formatted and added to context
    4. Combine chain generates answer from context
    5. Output includes answer, context, and input
    
    Args:
        vectorstore: Vector store containing embedded documents.
        llm: Language model for generating answers.
        search_kwargs: Optional search parameters (k, score_threshold, etc.).
    
    Returns:
        Runnable: Complete RAG chain that accepts {"input": str} and returns
            {"answer": str, "context": List[Document], "input": str}.
    
    Example:
        >>> vectorstore = InMemoryVectorStore(...)
        >>> llm = get_chat_model()
        >>> chain = build_rag_chain(vectorstore, llm)
        >>> result = chain.invoke({"input": "What is LangChain?"})
        >>> print(result["answer"])
    """
    # Configure retriever with search parameters
    # Default to top 3 most relevant documents
    if search_kwargs is None:
        search_kwargs = {"k": 3}
    
    # Create retriever from vector store
    # The retriever will perform similarity search when invoked
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    print(f"✓ Retriever configured with search parameters: {search_kwargs}")
    
    # Create the document combination chain
    combine_docs_chain = create_combine_docs_chain(llm)
    
    # Build complete retrieval chain
    # create_retrieval_chain automatically:
    # - Runs retriever to get relevant documents
    # - Passes documents as "context" to combine_docs_chain
    # - Returns dict with answer, context, and input
    # Source: libs/langchain/tests/unit_tests/chains/test_retrieval.py:16
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    print("✓ RAG chain created successfully")
    
    return rag_chain


def demonstrate_rag() -> None:
    """
    Main demonstration function showing complete RAG workflow.
    
    This function demonstrates:
    1. Document creation with sample LangChain content
    2. Embeddings model initialization with fallback handling
    3. Vector store creation and document indexing
    4. RAG chain construction
    5. Query execution with detailed output formatting
    6. Error handling for missing API keys
    
    Raises:
        ValueError: If required dependencies or API keys are missing.
    
    Example:
        >>> demonstrate_rag()
        ✓ Using OpenAI embeddings (text-embedding-ada-002)
        ✓ Created 5 sample documents
        ...
    """
    print("=" * 70)
    print("Retrieval-Augmented Generation (RAG) Example")
    print("=" * 70)
    print()
    
    # Step 1: Create sample documents
    # In production, load from PDFs, websites, databases, etc.
    print("Step 1: Creating sample documents...")
    documents = create_sample_documents()
    print(f"✓ Created {len(documents)} sample documents")
    print(f"  Topics covered: {', '.join(set(d.metadata['topic'] for d in documents))}")
    print()
    
    # Step 2: Initialize embeddings model
    # OpenAI embeddings if available, else fake embeddings for demo
    print("Step 2: Initializing embeddings model...")
    try:
        embeddings = get_embeddings_model()
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    print()
    
    # Step 3: Create in-memory vector store
    # This indexes all documents with their embeddings for semantic search
    print("Step 3: Creating vector store and indexing documents...")
    try:
        # InMemoryVectorStore.from_documents creates embeddings for all docs
        # Source: libs/core/tests/unit_tests/vectorstores/test_in_memory.py:89
        vectorstore = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=embeddings
        )
        print(f"✓ Indexed {len(documents)} documents in vector store")
    except Exception as e:
        print(f"✗ Failed to create vector store: {e}")
        sys.exit(1)
    print()
    
    # Step 4: Initialize chat model
    print("Step 4: Initializing chat model...")
    try:
        llm = get_chat_model()
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    print()
    
    # Step 5: Build RAG chain
    print("Step 5: Building RAG chain...")
    rag_chain = build_rag_chain(
        vectorstore=vectorstore,
        llm=llm,
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )
    print()
    
    # Step 6: Run sample queries
    print("Step 6: Running sample queries...")
    print("=" * 70)
    print()
    
    # Sample questions to demonstrate RAG
    questions = [
        "What is LangChain?",
        "How does LCEL work?",
        "What is RAG and how does it work?",
        "What are embeddings used for?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Query {i}: {question}")
        print("-" * 70)
        
        try:
            # Invoke RAG chain with question
            # Input: {"input": str}
            # Output: {"answer": str, "context": List[Document], "input": str}
            result = rag_chain.invoke({"input": question})
            
            # Display results
            print(f"\nAnswer:")
            print(f"  {result['answer']}")
            print(f"\nRetrieved Context ({len(result['context'])} documents):")
            for j, doc in enumerate(result['context'], 1):
                # Show first 100 chars of each retrieved document
                preview = doc.page_content[:100]
                if len(doc.page_content) > 100:
                    preview += "..."
                print(f"  [{j}] {preview}")
                print(f"      Source: {doc.metadata.get('source', 'unknown')}")
            
        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        print("=" * 70)
        print()
    
    print("✓ RAG demonstration complete!")
    print()
    print("Key Takeaways:")
    print("  • RAG retrieves relevant context before generating answers")
    print("  • Vector stores enable semantic search beyond keyword matching")
    print("  • create_retrieval_chain handles retrieval + generation workflow")
    print("  • Output includes answer, retrieved context, and original input")
    print("  • Easily swap InMemoryVectorStore with production vector DBs")


if __name__ == "__main__":
    """
    Entry point for standalone execution.
    
    Usage:
        # With OpenAI API key:
        export OPENAI_API_KEY='your-key-here'
        python retrieval_qa_chain.py
        
        # Or using .env file:
        echo "OPENAI_API_KEY=your-key-here" > .env
        python retrieval_qa_chain.py
    
    Expected output:
        - Document creation confirmation
        - Vector store indexing status
        - Sample Q&A demonstrating RAG
        - Retrieved context for each question
    """
    # Load environment variables from .env file if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # python-dotenv not installed, environment variables must be set manually
        pass
    
    try:
        demonstrate_rag()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

