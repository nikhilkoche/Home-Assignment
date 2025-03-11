from fastapi import FastAPI, File, UploadFile, APIRouter, Request, WebSocket, WebSocketDisconnect, status, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from app_config import APP_CONFIG
from pathlib import Path
from utils import PDFProcessor
from vectordb import PineconeDB
from fastapi.responses import JSONResponse
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from connection_manager import ConnectionManager
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from llm.session_history import get_session_history
from contextlib import asynccontextmanager
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.retrievers.document_compressors import EmbeddingsFilter
from llm.chat import question_answer_prompt, contextualize_q_prompt
import asyncio


app = FastAPI()
docs_dir = 'documents/'


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],   
)

manager = ConnectionManager()

def bot_creation(retriever: ContextualCompressionRetriever, 
                 llm: ChatOpenAI, 
                 contextualize_q_prompt: ChatPromptTemplate, 
                 qa_prompt: ChatPromptTemplate
                 ) -> RunnableWithMessageHistory:
    """
    
    Function to create the LangChain model for the chatbot.

    Args:
        retriever: The retriever model
        llm: The language model
        contextualize_q_prompt: The prompt for the contextualization model
        qa_prompt: The prompt for the question-answer model
    
    Returns:
        RunnableWithMessageHistory: The LangChain model
    
    """
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

async def get_ai_response(message: str, chain, session_id: str):
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
    print("Entered get_ai_response")

    async for chunk in chain.astream({"input": message}, config={"configurable": {"session_id": session_id}}):
        # print("Raw chunk received:", chunk)  

        if isinstance(chunk, dict) and "answer" in chunk:
            answer = chunk["answer"]

            
            if isinstance(answer, dict) or hasattr(answer, "get"):
                answer = answer.get("text", "")  
            elif isinstance(answer, list):  
                answer = "\n".join([str(a) for a in answer])  
            elif not isinstance(answer, str):
                answer = str(answer)  

            if answer.strip():  
                content += answer
                yield content


@asynccontextmanager
async def manage_connection(websocket: WebSocket, client_id: str):
    """
    
    Function to manage the WebSocket connection.

    Args:
        websocket: The WebSocket connection
        client_id: The client ID for the conversation
    
    Yields:
        tuple[str, str]: The session ID and unique ID for the client
    
    """
    session_id = None
    unique_id = None
    try:
        session_id, unique_id = await manager.connect(client_id, websocket)
        yield session_id, unique_id
    finally:
        if unique_id:
            await manager.disconnect(unique_id)


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Home page for the FastAPI app
    """
    current_dir = Path(__file__).parent
    with open(current_dir / 'index.html', 'r') as f:
        html_context = f.read()
    return html_context

@app.post("/upload_pdf")
def upload_pdf_file(file: UploadFile = File(...)) -> JSONResponse:
    """Handles PDF upload, processes it, and stores extracted text in Pinecone."""
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        # Save the uploaded PDF
        file_path = f"{docs_dir}/{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        file.file.close()

        # Process the PDF
        pdf_processor = PDFProcessor(file_path)
        documents = pdf_processor.process()

        # Upload extracted text chunks to Pinecone
        pc = PineconeDB(pinecone_api_key=APP_CONFIG.get('pinecone_key'),
                        openai_api_key=APP_CONFIG.get('openai_key'), 
                        index_name="project-j-index")
        document_ids = pc.add_documents(documents)

        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF processed and uploaded successfully.",
                "total_chunks": len(documents),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    
    WebSocket endpoint for the chatbot.

    Args:
        websocket: The WebSocket connection
        client_id: The client ID for the conversation
    """
    index_name = "project-j-index"
    pinecone_vectordb = PineconeVectorStore(index_name=index_name, 
                                            embedding=OpenAIEmbeddings(model="text-embedding-3-large"), 
                                            pinecone_api_key=APP_CONFIG['pinecone_key'])
    llm_bot = ChatOpenAI(model=APP_CONFIG['llm_model'],
                     temperature=APP_CONFIG['llm_temperature'], 
                     api_key=APP_CONFIG['openai_key'])
    async with manage_connection(websocket, client_id) as (session_id, unique_id):
        try:
            base_retriever = pinecone_vectordb.as_retriever(search_type="similarity",
                                                       search_kwargs={
                                                           "k": 20,
                                                       })
            embeddings_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(model="text-embedding-3-large"), similarity_threshold=0.2)
            retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=base_retriever,
            )
            conversational_rag_chain = bot_creation(retriever, llm_bot, contextualize_q_prompt, question_answer_prompt)
            await manager.send_json(unique_id, {
                "type": "stream",
                "content": "Hello! I'm here to help with the PDF that you have uploaded. Please ask any question you may have."
            })
            await manager.send_json(unique_id, {"type": "done", "content": ""})
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(), 
                        timeout=3600
                    )
                    async for text in get_ai_response(message, conversational_rag_chain, session_id):
                        await manager.send_json(unique_id, {
                            "type": "stream",
                            "content": text
                        })
                    await manager.send_json(unique_id, {"type": "done", "content": ""})
                except asyncio.TimeoutError:
                    await manager.send_json(unique_id, {
                        "type": "stream",
                        "content": "I'm sorry, I didn't receive a message for a while. Please try again."
                    })
                    await manager.send_json(unique_id, {"type": "done", "content": ""})
                    await manager.disconnect(unique_id, reason="Connection timeout")

        except WebSocketDisconnect as e:
            print(e)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    uvicorn.run(app, 
                host=APP_CONFIG['host'],
                port=APP_CONFIG['port'])
    