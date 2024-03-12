from typing import Any, Dict

from haystack import Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tqdm import tqdm

from rag_pipelines import BeirDataloader

prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:"""

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
sparse_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
sparse_pipeline.add_component(
    instance=HuggingFaceLocalGenerator(
        model="google/flan-t5-small",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
    ),
    name="llm",
)
sparse_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")


sparse_pipeline.connect("bm25_retriever", "prompt_builder.documents")
sparse_pipeline.connect("prompt_builder", "llm")
sparse_pipeline.connect("llm.replies", "answer_builder.replies")
sparse_pipeline.connect("bm25_retriever", "answer_builder.documents")


answers: Dict[str, Any] = {}

for query_id, query in tqdm(queries.items()):
    output = sparse_pipeline.run(
        {
            "bm25_retriever": {"query": query},
            "prompt_builder": {"question": query},
            "answer_builder": {"query": query},
        }
    )
    generated_answer = output["answer_builder"]["answers"][0]
    answers[query_id] = generated_answer
