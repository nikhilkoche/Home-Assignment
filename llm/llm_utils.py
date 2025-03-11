from collections.abc import AsyncGenerator

async def get_ai_response(message: str, chain, session_id: str) -> AsyncGenerator:
    """
    Function to get the streamed response from the AI model.

    Args:
        message: The message to send to the AI model
        chain: The LangChain model
        session_id: The session ID for the conversation (used for chat history context)

    Returns:
        AsyncGenerator: The generator for the AI model response
    """
    content: str = ""
    async for chunk in chain.astream({"input": message}, config={"configurable": {"session_id": session_id}}):
        if 'answer' in chunk and not chunk['answer'] == "":
            content += chunk['answer']
            yield content