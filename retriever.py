from typing import List, Dict
from vector_store import VectorStore
from embedding_manager import EmbeddingManager

class RAGRetriever:
    """
    Handles query-based retrieval from the vector store
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        subject: str = None,
        year: int = None,
        session: str = None,
        paper_id: str = None,
        top_k: int = 10,
        score_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            subject: Subject filter
            year: Optional year filter
            session: Optional session filter
            paper_id: Optional exact paper filter
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"🔍 Retrieving documents for query: {query}")
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Construct proper ChromaDB 'where' filter
        filters = []

        if subject:
            filters.append({"subject": {"$eq": subject}})
        if year:
            filters.append({"year": {"$eq": year}})
        if session:
            filters.append({"session": {"$eq": session}})
        
        # Correct way for ChromaDB: exactly one top-level operator
        if not filters:
            where_clause = None
        elif len(filters) == 1:
            # Only one filter, pass as is
            where_clause = filters[0]
        else:
            # Multiple filters → combine with $and
            where_clause = {"$and": filters}
        
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause
            )

            retrieved_docs = []
            if results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, content, metadata, distance) in enumerate(
                    zip(ids, docs, metadatas, distances)
                ):
                    similarity_score = 1 / (1 + distance)
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": content,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents after filtering")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
