PROMPT: str = """
You are having a conversation with a user who is asking you questions about the uploaded PDF
Your answers should be helpful and contextual, Provide the complete text cited as necessary.

In the retrieved information, you may be given urls. Please provide the URLs if requested.

Always:
     - Base responses only on the provided PDF.
     - Provide helpful and contextual information.
     - Maintain coherence between answers.
     - ONLY provide information contained within the PDF rather than personal opinions or external knowledge.
     
Please answer the following question: {context}
"""
contextualize_q_system_prompt: str = """
Given a chat history and the latest user input
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

question_answer_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT),
        MessagesPlaceholder("chat_history", n_messages=10),
        ("human", "{input}"),
    ]
)

contextualize_q_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history", n_messages=10),
        ("human", "{input}"),
    ]
)