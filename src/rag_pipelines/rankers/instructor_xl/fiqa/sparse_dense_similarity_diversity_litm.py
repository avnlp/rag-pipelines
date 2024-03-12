from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import (
    LostInTheMiddleRanker,
    SentenceTransformersDiversityRanker,
    TransformersSimilarityRanker,
)
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack_integrations.components.embedders.instructor_embedders import InstructorTextEmbedder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from tqdm import tqdm

from rag_pipelines import BeirDataloader, BeirEvaluator

dataset = "fiqa"
data_loader = BeirDataloader(dataset)
data_loader.download_and_unzip()
corpus, queries, qrels = data_loader.load()

documents_corp = [
    Document(
        content=text_dict["text"],
        meta={"corpus_id": str(corpus_id), "title": text_dict["title"]},
    )
    for corpus_id, text_dict in corpus.items()
]

sparse_document_store = InMemoryDocumentStore()
sparse_document_store.write_documents(documents_corp)

dense_document_store = PineconeDocumentStore(
    api_key=Secret.from_env_var("PINECONE_API_KEY"),
    environment="gcp-starter",
    index="fiqa",
    namespace="default",
    dimension=768,
)

dense_retriever = PineconeEmbeddingRetriever(document_store=dense_document_store, top_k=10)
query_instruction = "Represent the financial question for retrieving supporting documents:"
text_embedder = InstructorTextEmbedder(
    model="hkunlp/instructor-xl",
    instruction=query_instruction,
)

sparse_retriever = InMemoryBM25Retriever(document_store=sparse_document_store, top_k=10)

joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")

similarity_ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-large", top_k=10)
diversity_ranker = SentenceTransformersDiversityRanker(model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k=10)
litm_ranker = LostInTheMiddleRanker(top_k=10)

hybrid_pipeline = Pipeline()

hybrid_pipeline.add_component(instance=sparse_retriever, name="bm25_retriever")
hybrid_pipeline.add_component(
    instance=text_embedder,
    name="text_embedder",
)
hybrid_pipeline.add_component(instance=dense_retriever, name="embedding_retriever")
hybrid_pipeline.add_component(instance=joiner, name="joiner")
hybrid_pipeline.add_component(instance=similarity_ranker, name="similarity_ranker")
hybrid_pipeline.add_component(instance=diversity_ranker, name="diversity_ranker")
hybrid_pipeline.add_component(instance=litm_ranker, name="litm_ranker")

hybrid_pipeline.connect("bm25_retriever", "joiner")
hybrid_pipeline.connect("text_embedder", "embedding_retriever")
hybrid_pipeline.connect("embedding_retriever", "joiner")
hybrid_pipeline.connect("joiner.documents", "similarity_ranker.documents")
hybrid_pipeline.connect("similarity_ranker.documents", "diversity_ranker.documents")
hybrid_pipeline.connect("diversity_ranker.documents", "litm_ranker.documents")

result_qrels_all = {}

for query_id, query in tqdm(queries.items()):
    output = hybrid_pipeline.run(
        {
            "bm25_retriever": {"query": query},
            "text_embedder": {"text": query},
            "similarity_ranker": {"query": query},
            "diversity_ranker": {"query": query},
        }
    )
    output_docs = output["litm_ranker"]["documents"]
    doc_qrels = {}
    for doc in output_docs:
        doc_qrels[doc.meta["corpus_id"]] = doc.score
    result_qrels_all[query_id] = doc_qrels
evaluator = BeirEvaluator(qrels, result_qrels_all, [3, 5, 7, 10])
ndcg, _map, recall, precision = evaluator.evaluate()
