from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Tuple, Union
from pinecone import Pinecone as PineconeClient, ServerlessSpec

class PineconeDB:
    def __init__(
        self, 
        pinecone_api_key: str,
        openai_api_key: str,
        index_name: str,
        embedding_model: str = "text-embedding-3-large"
    ):
        
        # Initialize Pinecone
        self.pc = PineconeClient(api_key=pinecone_api_key)
        
        # Set up OpenAI embeddings
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Create index if it doesn't exist
        dimension: int = 1536 if embedding_model in ["text-embedding-3-small", "text-embedding-ada-002"] else 3072
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        # Get the index
        self.index = self.pc.Index(index_name)
        
        # Initialize LangChain's Pinecone integration
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"
        )
    
    def is_document_processed(self, pc) -> bool:
        
        stats = pc.index.describe_index_stats()
        return stats.get('total_vector_count') > 0  
    
    def add_documents(self, documents: List[Document], filename: str) -> List[str]:
    
        texts = [doc.page_content for doc in documents]
        metadatas = [{"text": doc.page_content,  "source": filename,**doc.metadata} for doc in documents]
        
        return self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )
    
    def similarity_search(
        self,
        query: Union[str, Document],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Document]:

        if isinstance(query, Document):
            query = query.page_content
            
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
            include_metadata=include_metadata
        )
    
    def similarity_search_with_score(
        self,
        query: Union[str, Document],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
    
        if isinstance(query, Document):
            query = query.page_content
            
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def delete_documents(self, ids: List[str]) -> None:
    
        self.vectorstore.delete(ids)