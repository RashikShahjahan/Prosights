import logging
from celery import Celery
from rag import create_rag_chain, invoke_rag, get_session_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Celery app
app = Celery('tasks',
             broker='amqp://guest@localhost//',
             backend='redis://localhost:6379/0')

# Create RAG chain
rag_chain = create_rag_chain('data/', 'gpt-4o')

@app.task(bind=True, max_retries=3)
def question_task(self, conversation_id: str, question: str):
    """
    Process a question using RAG and update session history.

    Args:
        conversation_id (str): Unique identifier for the conversation.
        question (str): The user's question.

    Returns:
        str: The generated answer.
    """
    logger.info(f"Processing question for conversation {conversation_id}: {question}")
    try:
        # Generate the answer
        answer = invoke_rag(rag_chain, question, conversation_id)

        # Update the session history
        history = get_session_history(conversation_id)
        history.add_user_message(question)
        history.add_ai_message(answer)

        logger.info(f"Answer generated for conversation {conversation_id}: {answer}")
        return answer
    except Exception as exc:
        logger.error(f"Error processing question for conversation {conversation_id}: {exc}")
        raise self.retry(exc=exc, countdown=2**self.request.retries)