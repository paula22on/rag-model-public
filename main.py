# RAG Model
import glob

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq

from config import GROQ_API_KEY
from utils import extract_text_from_pdf, print_response

print("This is the start of your RAG model")
# Steps

folder_path = "documents/input_documents"
input_files_path = glob.glob(f"{folder_path}/*.pdf")
all_documents = []

for input_file in input_files_path:
    print(f"Document name: {input_file}")

    # 1. Read document
    print("Parsing document...")
    text = extract_text_from_pdf(input_file)

    # 2. Split documents into chunks
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    text_splitted = text_splitter.split_text(text)
    documents = text_splitter.create_documents(text_splitted)

    # Add documents to the list
    all_documents.extend(documents)

# 3. Create embeddings for each text chunk
print("Creating embeddings...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 4. Load into vector database
print("Loading embeddings into vector database...")
qdrant = Qdrant.from_documents(
    all_documents,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)

# Define prompt
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Answer the question and provide additional helpful information,
based on the pieces of information, if applicable. Be succinct.

Responses should be properly formatted to be easily read.
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Initialize LLM
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# Initialize retriever for reranking
retriever = qdrant.as_retriever(search_kwargs={"k": 5})

# compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


def query(query):
    # 5. Create embeddings for the question and retrieve similar chunks from the vector database
    print("Create query embeddings...")
    # found_docs = qdrant.similarity_search_with_score(query)

    # for doc, score in found_docs:
    #     print(f"text: {doc.page_content[:256]}\n")
    #     print(f"score: {score}")
    #     print("-" * 80)
    #     print()

    reranked_docs = compression_retriever.invoke(query)
    len(reranked_docs)

    for doc in reranked_docs:
        print(f"id: {doc.metadata['_id']}\n")
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {doc.metadata['relevance_score']}")
        print("-" * 80)
        print()

    # 7. Combine context and question in a prompt and generate response
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )

    response = qa.invoke(query)

    print_response(response)
    return response
