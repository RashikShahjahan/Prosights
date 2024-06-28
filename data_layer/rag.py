import logging
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import RedisChatMessageHistory
logger = logging.getLogger(__name__)

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """Get or create a Redis-based chat message history for a given session ID."""
    return RedisChatMessageHistory(session_id)

def create_rag_chain(db_path: str, model_name: str, temperature: float = 0.0) -> RunnableWithMessageHistory:
    """
    Create a RAG chain with history-aware retriever and question-answering capabilities.

    Args:
        db_path (str): Path to the Chroma database.
        model_name (str): Name of the OpenAI model to use.
        temperature (float, optional): Temperature for the language model. Defaults to 0.0.

    Returns:
        RunnableWithMessageHistory: A conversational RAG chain.
    """
    vectordb = Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=os.getenv('OPENAI_API_KEY'))

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def invoke_rag(rag_chain: RunnableWithMessageHistory, input: str, session_id: str) -> str:
    """
    Invoke the RAG chain with a given input and session ID.

    Args:
        rag_chain (RunnableWithMessageHistory): The RAG chain to invoke.
        input (str): The user's input question.
        session_id (str): The session ID for message history.

    Returns:
        str: The generated answer.
    """
    logger.debug("Inside invoke_rag")

    result = rag_chain.invoke(
        {"input": input},
        config={"configurable": {"session_id": session_id}},
    )

    logger.debug(f"RAG result: {result}")
    return result["answer"]