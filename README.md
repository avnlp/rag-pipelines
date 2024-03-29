# RAG Pipelines

Retrieval Augmented Generation (RAG) is a method for generating text using additional information fetched from an external data source. Providing relevant documents to the model can greatly increase the accuracy of the generated response.

A RAG pipeline can be tuned in many ways to give more relevant answers. One important way is to improve the relevance of the retrieved context which is input to the LLM. This ensures that the generated answers are coherent and consistent with the context in the original documents.

We have done a detailed analysis of RAG pipelines for Dense and Hybrid Retrieval. For dense retrieval the INSTRUCTOR-XL and all-mpnet-base-v2 models were used.
The BM25 retrieval was used for sparse retrieval in the hybrid pipelines.

<img src="plots/pipelines_taxonomy.png" alt="RAG Pipelines Taxonomy" align="middle" width="600" height="300">


## Performance Evaluation of Rankers and RRF Techniques for Retrieval Pipelines

**Paper:** [Performance Evaluation of Rankers and RRF Techniques for Retrieval Pipelines](https://github.com/avnlp/rag-pipelines/blob/main/paper/rankers_rrf.pdf)

In the intricate world of LFQA and RAG, making the most of the LLM’s context window is paramount. Any wasted space or repetitive content limits the depth and breadth of the answers we can extract and generate. It’s a delicate balancing act to lay out the content of the context window appropriately. 

With the addition of three rankers, viz., DiversityRanker, LostInTheMiddleRanker, Similarity rankers and RRF techniques, we aim to address these challenges and improve the answers generated by the LFQA/RAG pipelines. We have done a comparative study of adding different combinations of rankers in a Retrieval pipeline and evaluated the results on four metrics, viz., Normalized Discounted Cumulative Gain (NDCG), Mean Average Precision (MAP), Recall and Precision .

The following rankers were used:

- Diversity Ranker: The Diversity Ranker enhances the diversity of the paragraphs selected for the context window.

- Lost In The Middle Ranker: The Lost In The Middle Ranker optimizes the layout of the selected documents in the LLM’s context window.

- Transformers Similarity Ranker: The Transformers Similarity Ranker ranks Documents based on how similar they are to the query. It uses a pre-trained cross-encoder model to embed both the query and the Documents. It then compares the embeddings to determine how similar they are.

In our study, we consider the following cases of retrieval:

- Dense Retrieval

<img src="plots/rankers_dense_pipeline.png" alt="Dense Pipeline with Rankers" align="middle" width="550" height="100">


- Hybrid Retrieval

To combine the results for Hybrid retrieval, Reciprocal Rank Fusion (RRF) was used.

<img src="plots/rankers_hybrid_pipeline.png" alt="Hybrid Pipeline with Rankers" align="middle" width="820" height="230">


## Usage

To run the retreival pipelines, you will need to clone this repository and install the required libraries.


1. Install the `rag-pipelines` package:

```bash
git clone https://github.com/avnlp/rag-pipelines
cd rag-pipelines
pip install -e .
```

2. To add the data to an index in Pinecone using the INSTRUCTOR-XL embedding model:

```python 
cd src/rag_pipelines/indexing_pipeline/fiqa
python pinecone_instructor_index.py
```

3. To run a specific pipeline you will have to go that file path and then run the file.
For example, running the dense retreival pipeline using the INSTRUCTOR-XL embedding model:

```python 
cd src/rag_pipelines/rankers/instructor_xl/fiqa/
python dense.py
```

## License

The source files are distributed under the [MIT License](https://github.com/avnlp/rag-pipelines/blob/main/LICENSE).
