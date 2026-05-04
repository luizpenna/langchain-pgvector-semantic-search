import os
import argparse
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search inside documents in PostgreSQL + pgVector."
    )
   
    parser.add_argument(
        "--provider",
        choices=["openai", "google"],
        help="Embedding provider. Falls back to EMBEDDING_PROVIDER from .env",
    )

    parser.add_argument(
        "--message",
        help="The search message to query the vector store. Provide the message as a command line argument. Example: python search.py --message 'What is the capital of France?'",
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

def search (search_query: str,  args: argparse.Namespace) -> list[dict[str, object]]:

    load_dotenv()

    logs = args.logs or (os.getenv("LOG") == "1")

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


    # Define embeddings
    match provider:
        case 'google':
            model = os.getenv('GOOGLE_EMBEDDING_MODEL', 'models/embedding-001')
            embeddings = GoogleGenerativeAIEmbeddings(model=model)
        case 'openai' | _:
            model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')        
            embeddings = OpenAIEmbeddings(model=model)

    # create vector store instance
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=f"{collection_name}_{provider}",  # Append provider to collection name to avoid conflicts if using multiple providers
        connection=connection_string,
        use_jsonb=True,
    )
    
    docs = vectorstore.similarity_search_with_score(search_query, k=10)
    if logs:
        print(f"Search query: {search_query}")
        print(f"Found {len(docs)} similar documents in collection: {collection_name}_{provider}")
        for i, (doc, score) in enumerate(docs, start=1):
            print(f"Content: {doc.page_content[:100]}..., Score: {score:.2f}\n")

    results = [{"page_content": doc.page_content, "score": float(score)} for doc, score in docs]
    return results

if __name__ == "__main__":

    args = parse_args()

    message = args.message;
    if not message:
        raise RuntimeError("No search message provided - provide as command line argument")

    search(message, args)            