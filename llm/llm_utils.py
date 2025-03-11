from collections.abc import AsyncGenerator

async def get_ai_response(message: str, chain, session_id: str) -> AsyncGenerator:
  
    content: str = ""
    async for chunk in chain.astream({"input": message}, config={"configurable": {"session_id": session_id}}):
        if 'answer' in chunk and not chunk['answer'] == "":
            content += chunk['answer']
            yield content