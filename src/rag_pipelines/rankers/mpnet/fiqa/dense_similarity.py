from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.utils import Secret
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
dense_document_store = PineconeDocumentStore(
    api_key=Secret.from_env_var("PINECONE_API_KEY"),
    environment="gcp-starter",
    index="fiqa",
    namespace="default",
    dimension=768,
)

dense_retriever = PineconeEmbeddingRetriever(document_store=dense_document_store, top_k=10)
text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-mpnet-base-v2",
)
similarity_ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-large", top_k=10)

dense_pipeline = Pipeline()
dense_pipeline.add_component(
    instance=text_embedder,
    name="text_embedder",
)
dense_pipeline.add_component(instance=dense_retriever, name="embedding_retriever")
dense_pipeline.add_component(instance=similarity_ranker, name="similarity_ranker")

dense_pipeline.connect("text_embedder", "embedding_retriever")
dense_pipeline.connect("embedding_retriever.documents", "similarity_ranker.documents")

result_qrels_all = {}

for query_id, query in tqdm(queries.items()):
    output = dense_pipeline.run({"text_embedder": {"text": query}, "similarity_ranker": {"query": query}})
    output_docs = output["similarity_ranker"]["documents"]
    doc_qrels = {}
    for doc in output_docs:
        doc_qrels[doc.meta["corpus_id"]] = doc.score
    result_qrels_all[query_id] = doc_qrels

evaluator = BeirEvaluator(qrels, result_qrels_all, [3, 5, 7, 10])
ndcg, _map, recall, precision = evaluator.evaluate()
