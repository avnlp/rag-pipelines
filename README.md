# RAG Pipelines

Earnings calls are a critical source of information for institutional investors, helping them make better investment decisions. The transcripts of these calls are voluminous, generated every quarter, and difficult to parse and correlate. Hence, extracting actionable information from multiple transcripts is extremely crucial.

We have built a pipeline to perform open-ended question-answering on earnings call transcripts using Generative Large Language Models (LLMs). The pipeline retrieves data from outside the model using Semantic Search and augments the prompts by adding the retrieved data in-context. LLMs effectively use the new information provided to them in the prompts to generate relevant text.

- RAG Pipelines were built on financial datasets consisting of earnings call transcripts and SEC Filings (10-K, 10-Q, 8-K) for question answering.
- The pipelines use Hybrid Retrieval with the Instructor-XL embedding model and the SPLADE sparse embedding models.
- The dataloaders for the earnings call transcripts and SEC filings (10-K, 10-Q, 8-K) datasets were used from the `dataloaders` repository.
- The RAG pipelines include CRAG and Self-RAG variants built on financial datasets for robust question-answering tasks.
- The Hybrid Retrieval pipeline combines Instructor-XL dense embeddings with SPLADE sparse embedding.
- Using the Groq API the Llama-3.1 8B model was used for response generation.
- For Response Evaluation the DeepEval metrics Answer Relevancy, Faithfulness, Hallucination and Summarization were used in pipelines.
- For Retrieval Evaluation the metrics Contextual Precision, Contextual Recall, Contextual Relevancy were used in pipelines.
- DeepEval metrics were utilized for scoring retrieval and response evaluation, with results being logged using weave.
- Separate scripts enable the creation of embeddings, Pinecone indexing, and the execution of Hybrid RAG, CRAG, and Self-RAG pipelines.
- Using the Groq API the Llama-3.1 30B model was used for evaluation.
- The full traces of the pipelines were logged using Weave.

## Data

### Structure of Earnings Calls Transcript

**Corporate Participants and Conference Call Participants Sections**: The host of the call introduces the company and the participants. This section contains a list of speakers, their designations and the name of the company they represent. It contains names of the analysts and
investors who are participating in the call. It also contains the names of the company’s executives who are participating in the call, such as the CEO, CFO, and other senior executives.

**Presentation Section**: It provides a comprehensive overview of the company’s financial perfor- mance and strategic direction. The company presents its financial results for the quarter or year, including revenue, earnings per share, and other key metrics.

**Question-Answer Section**: The host opens the call to questions from analysts and investors. This segment can provide additional insights into the company’s performance and future prospects. This section contains the questions that were asked by analysts and investors during the call, as well
as the responses from the company’s executives.

We have used the earnings call dataset [Earnings Calls QA](https://huggingface.co/datasets/lamini/earnings-calls-qa)
The dataset contains the following columns:

- **Question**: The query or question posed during the earnings call.
- **Answer**: The response to the query, provided during the call.
- **Context**: A snippet of the transcript containing relevant information.
- **Metadata**: Additional details such as speaker information, timestamps, or topics.

## Self RAG on Earnings Calls

- Self-RAG introduces self-reflection into traditional RAG pipelines. It allows the system to evaluate its own intermediate outputs, assess their quality, and decide whether further refinement is necessary before generating a final response.

- An arbitrary LM is trained in an end-to-end manner to learn to reflect on its own generation process given a task input by generating both task output and intermittent special tokens (i.e., reflection tokens). Reflection tokens are categorized into retrieval and critique tokens to
indicate the need for retrieval and its generation quality respectively. SELF-RAG first determines if augmenting the continued generation with retrieved passages would be helpful.

- If so, it outputs a retrieval token that calls a retriever model on demand. Subsequently, SELF-RAG concurrently processes multiple retrieved passages, evaluating their relevance and then generating corresponding task outputs. It then generates critique tokens to criticize its own output and choose best one in terms of factuality and overall quality.

We have integrated the Self RAG framework to work with the earnings calls data.
The pipeline constructs a workflow graph to define the order and conditions for various processing stages. Nodes represent actions like:

- Retrieve: Fetch documents using the retriever.
- Grade Documents: Assess relevance using the grader.
- Transform Query: Modify and optimize queries if needed.
- Web Search: Search for additional context online.
- Generate: Produce the final output using the generator.

Conditional transitions are added to decide if the workflow should optimize queries or proceed directly to response generation. The graph is compiled for execution.

When a query is received, the pipeline builds the workflow graph and sets up the Weave tracer for monitoring. The compiled graph processes the query through the defined nodes and transitions, generating a final response based on the retrieved and processed information.

### Query Transfomer

The query transformer is used for transforming the input query into a similar query for better retrieval performance.
This is achieved by using a Langchain pipeline to generate a new query from the original query.

### Retreival Evaluator

The Retreival evaluator is used for evaluating the quality of the retrieved documents using an LLM.
The pipeline iterates over the retrieved documents and evaluates their relevance using an LLM.

### WebSearch

The Websearch component searching the web for relevant documents based on the query. It uses the DuckDuckGo API for external web searches.

## License

The source files are distributed under the [MIT License](https://github.com/avnlp/rag-pipelines/blob/main/LICENSE).
