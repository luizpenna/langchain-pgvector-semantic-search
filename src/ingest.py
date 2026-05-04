import os
import sys
import uuid
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
from langchain_core.documents import Document


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into PostgreSQL + pgVector."
    )

    parser.add_argument(
        "--pdf-path",
        dest="pdf_path",
        help="Path to the PDF file. Falls back to PDF_PATH from .env",
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "google"],
        help="Embedding provider. Falls back to EMBEDDING_PROVIDER from .env",
    )

    parser.add_argument(
        "--collection-name",
        dest="collection_name",
        help="Name of the collection to store document chunks. Falls back to COLLECTION_NAME from .env",
    )

    parser.add_argument(
        "--logs",
        action="store_true",
        help="Enable debug logs",
    )   

    return parser.parse_args()

def ingest_pdf ():    

    load_dotenv()

    args = parse_args()

    logs = args.logs or (os.getenv("LOG") == "1")

    # Get PDF path from command line argument or .env variable
    pdf_path = args.pdf_path or os.getenv("PDF_PATH")
    if not pdf_path:
        raise RuntimeError("No PDF path provided - set PDF_PATH environment variable or provide as command line argument")

    # Get collection name from command line argument or .env variable
    collection_name = args.collection_name or os.getenv("PG_VECTOR_COLLECTION_NAME")
    if not collection_name:
        raise RuntimeError("No collection name provided - set PG_VECTOR_COLLECTION_NAME environment variable or provide as command line argument")

    # Get embedding provider from environment variable or use default (openai)
    provider = (args.provider or os.getenv("EMBEDDING_PROVIDER", "openai")).strip().lower()
    if not provider:
        raise RuntimeError("No embedding provider specified - set EMBEDDING_PROVIDER environment variable or provide as command line argument (openai or google)")
    elif provider not in ['openai', 'google']:
        raise RuntimeError(f"Unsupported embedding provider: {provider}. Supported providers are: openai, google")

    # Get database connection string from environment variable
    connection_string = os.getenv('DATABASE_URL')
    if not connection_string:
        raise RuntimeError("DATABASE_URL not set - set DATABASE_URL environment variable to connect to your PostgreSQL database")

    #Print configuration for debugging
    if logs:
        print(f"PDF Path: {pdf_path}")
        print(f"Embedding Provider: {provider}") 

    # Define PDF loader and load the document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Define text splitter and split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Print number of chunks created for debugging
    if logs:
        print(f"Document loaded and split into {len(chunks)} chunks")
        print(f"Sample chunk content: {chunks[0].page_content[:200]}...")  # Print first 200 characters of the first chunk for debugging        

    enriched = [
        Document(
            # Generate a unique ID for each chunk based on its content and metadata and the provider to ensure uniqueness across different providers
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(d.page_content) + str(d.metadata) + provider)), 
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in chunks
    ]

    # Print generated IDs and how many chunks were enriched for debugging
    if logs:
        print(f"Enriched {len(enriched)} chunks with metadata")
        print(f"Generated UID {[doc.id for doc in enriched]}")
            

    # Define embeddings and create vector store from document 
    match provider:
        case 'google':
            model = os.getenv('GOOGLE_EMBEDDING_MODEL', 'models/embedding-001')
            embeddings = GoogleGenerativeAIEmbeddings(model=model)
        case 'openai' | _:
            model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')        
            embeddings = OpenAIEmbeddings(model=model)

    if logs:
        print(f"Using embedding model: {model} from provider: {provider}")

    store = PGVector(
        embeddings=embeddings,
        collection_name=f"{collection_name}_{provider}",  # Append provider to collection name to avoid conflicts if using multiple providers
        connection=connection_string,
        use_jsonb=True,                
    )
    
    inserted_ids = store.add_documents(documents=enriched)
    if logs:
        print(f"Inserted {len(inserted_ids)} document chunks into PGVector collection: {store.collection_name}")


if __name__ == "__main__":
    ingest_pdf()