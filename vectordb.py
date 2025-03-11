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
        """
        Initialize Pinecone database with LangChain Document support.
        
        Args:
            pinecone_api_key: Your Pinecone API key
            openai_api_key: Your OpenAI API key
            index_name: Name of the index to use
            embedding_model: OpenAI embedding model to use
        """
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
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add LangChain Document objects to the vector database.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [{"text": doc.page_content, **doc.metadata} for doc in documents]
        
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
        """
        Search for similar documents.
        
        Args:
            query: Query text or Document object
            k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of similar Document objects
        """
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
        """
        Search for similar documents and return similarity scores.
        
        Args:
            query: Query text or Document object
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples (Document, score)
        """
        if isinstance(query, Document):
            query = query.page_content
            
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the database.
        
        Args:
            ids: List of document IDs to delete
        """
        self.vectorstore.delete(ids)