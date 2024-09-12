# RAG Model (Retrieval-Augmented Generation)

This project implements a Retrieval-Augmented Generation (RAG) model using the LangChain framework. It processes documents, stores them in a vector database, and uses a Large Language Model (LLM) to answer queries based on the retrieved information. A Flask-based frontend has also been developed to provide a more user-friendly way of interacting with the model.

## Features

- Document Parsing: Extract text from PDF files and process them.
- Text Chunking: Split documents into smaller chunks for efficient retrieval.
- Embeddings Creation: Generate embeddings for document chunks using FastEmbedEmbeddings.
- Vector Database Storage: Store document embeddings in Qdrant, an in-memory vector database.
- Contextual Retrieval: Retrieve relevant document chunks based on query using a retrieval model.
- Reranking: Re-rank retrieved documents using FlashrankRerank to improve the relevance of results.
- LLM Integration: Use the Groq-powered llama3-70b-8192 model to generate responses from the retrieved information.
- Flask Frontend: A user-friendly web interface built using Flask allows you to ask questions and interact with the RAG model.

## Prerequisites

- Python 3.9.19
- Necessary dependencies are listed in the requirements.txt file. You can install them using:
  ```
  pip install -r requirements.txt
  ```

## Configuration

1. API Key: Ensure you have a valid GROQ_API_KEY from Groq. You can configure it in the config.py file:

```
GROQ_API_KEY = "your_groq_api_key_here"

```

2. Document Input: Add the PDF files you want to process into the documents/input_documents folder.

## How It Works

Steps:

1. Document Parsing: The script reads PDF files from the specified folder and extracts the text.
2. Text Splitting: The extracted text is chunked into smaller pieces for efficient processing and storage.
3. Embeddings Creation: The chunks are passed through FastEmbedEmbeddings to generate embeddings.
   4.Vector Database: The embeddings are stored in a Qdrant vector database for fast retrieval.
4. Query Processing: When a query is made, embeddings are generated for the query, and the most relevant document chunks are retrieved.
5. Re-ranking: The retrieved chunks are reranked to ensure the most relevant chunks are provided to the model.
6. Answer Generation: The relevant context is passed to the LLM (llama3-70b-8192 via Groq) to generate an answer.

## Flask Frontend

A Flask-based web interface is also included in this project. It allows you to ask questions to the model and view responses through a simple web page.

Flask Endpoints

1. Home Page:
   - Route: /
   - Description: Renders the main chatbot interface.
2. Ask a Question:
   - Route: /answer
   - Method: GET
   - Params: query (the question you want to ask)
   - Response: Returns the model’s generated response for the given question.
3. Get Available Files:
   - Route: /get-files
   - Method: GET
   - Description: Retrieves a list of the document files available for querying.
4. Get Sample Questions:
   - Route: /get-questions
   - Method: GET
   - Description: Fetches pre-defined sample questions from a JSON file.

## Running the Flask App

1. Start the Flask server:

```
python app.py

```

2. Open a browser and navigate to http://127.0.0.1:5000/ to interact with the chatbot.

## Example

### Querying the Model via Flask

To interact with the model, simply go to the home page and type a question into the input field. The query will be sent to the model, and the response will be displayed on the screen.

### Querying Directly in Python

You can also query the model directly by calling the query function in the Python script:

```
response = query("What are the key points from the document?")

```

The response will be generated based on the document's content.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
