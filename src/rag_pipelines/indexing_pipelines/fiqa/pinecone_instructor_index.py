from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.components.embedders.instructor_embedders import InstructorDocumentEmbedder
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from rag_pipelines import BeirDataloader

dataset = "fiqa"
data_loader = BeirDataloader(dataset)
data_loader.download_and_unzip()
corpus, queries, qrels = data_loader.load()

documents = [
    Document(
        content=text_dict["text"],
        meta={"corpus_id": str(corpus_id), "title": text_dict["title"]},
    )
    for corpus_id, text_dict in corpus.items()
]

document_store = PineconeDocumentStore(
    api_key=Secret.from_env_var("PINECONE_API_KEY"),
    environment="gcp-starter",
    index="fiqa",
    namespace="default",
    dimension=768,
)

doc_instruction = "Represent the financial document for retrieval:"

embedder = InstructorDocumentEmbedder(model="hkunlp/instructor-xl", instruction=doc_instruction)
embedder.warm_up()

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", DocumentWriter(document_store))
indexing_pipeline.connect("embedder", "writer")

indexing_pipeline.run({"embedder": {"documents": documents}})
