import os
import sys
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()
DEBUG = os.getenv("DEBUG") == "1"

def ingest_pdf ():    

    # Get PDF path from command line argument or .env variable
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv('PDF_PATH')
    if not pdf_path:
        raise RuntimeError("No PDF path provided - set PDF_PATH environment variable or provide as command line argument")

    # Get embedding provider from environment variable or use default (openai)
    provider = sys.argv[2] if len(sys.argv) > 2 else os.getenv('EMBEDDING_PROVIDER', 'openai')
    if not provider:
        raise RuntimeError("No embedding provider specified - set EMBEDDING_PROVIDER environment variable or provide as command line argument (openai or google)")
    elif provider not in ['openai', 'google']:
        raise RuntimeError(f"Unsupported embedding provider: {provider}. Supported providers are: openai, google")

    # Get database connection string from environment variable
    connection_string = os.getenv('DATABASE_URL')
    if not connection_string:
        raise RuntimeError("DATABASE_URL not set - set DATABASE_URL environment variable to connect to your PostgreSQL database")

    #Print configuration for debugging
    if DEBUG:
        print(f"PDF Path: {pdf_path}")
        print(f"Embedding Provider: {provider}") 

    # Define PDF loader and load the document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Define text splitter and split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Print number of chunks created for debugging
    if DEBUG:
        print(f"Document loaded and split into {len(chunks)} chunks")        

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in chunks
    ]

    if DEBUG:
        print(f"Enriched {len(enriched)} chunks with metadata")

    # Use metadata to generate UID for each document chunk.
    ids = []
    for d in enriched:
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(d.page_content) + str(d.metadata)))
        ids.append(uid)
        
    if DEBUG:
        print(f"Generated {len(ids)} unique IDs for document chunks")
        print(f"Generated UID {ids}")

    # Define embeddings and create vector store from document 
    match provider:
        case 'google':
            model = os.getenv('GOOGLE_EMBEDDING_MODEL', 'models/embedding-001')
            embeddings = GoogleGenerativeAIEmbeddings(model=model)
        case 'openai' | _:
            model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')        
            embeddings = OpenAIEmbeddings(model=model)

    if DEBUG:
        print(f"Using embedding model: {model} from provider: {provider}")

    
    store = PGVector(
        embeddings=embeddings,
        collection_name="my_document_chunks_collection",  
        connection=connection_string,
        use_jsonb=True,
    )

    store.add_documents(documents=enriched, ids=ids)


if __name__ == "__main__":
    ingest_pdf()