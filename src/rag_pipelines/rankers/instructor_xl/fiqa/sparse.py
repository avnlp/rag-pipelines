from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
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

sparse_retriever = InMemoryBM25Retriever(document_store=sparse_document_store, top_k=10)

sparse_pipeline = Pipeline()
sparse_pipeline.add_component(instance=sparse_retriever, name="bm25_retriever")

result_qrels_all = {}

for query_id, query in tqdm(queries.items()):
    output_docs = sparse_pipeline.run({"bm25_retriever": {"query": query}})["bm25_retriever"]["documents"]
    doc_qrels = {doc.meta["corpus_id"]: doc.score for doc in output_docs}
    result_qrels_all[query_id] = doc_qrels

evaluator = BeirEvaluator(qrels, result_qrels_all, [3, 5, 7, 10])
ndcg, _map, recall, precision = evaluator.evaluate()
