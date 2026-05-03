# LangChain + pgVector Semantic Search (CLI)

  PDF ingestion and semantic search using LangChain, PostgreSQL, and pgVector via command line.

  ## Objective

  This project implements:

  1. PDF ingestion.
  2. Chunk-based embedding generation.
  3. Vector storage in PostgreSQL with pgVector.
  4. Semantic retrieval with `k=10`.
  5. CLI answers based **only** on retrieved context.

  ## Tech Stack

  - Python
  - LangChain
  - PostgreSQL + pgVector
  - Docker Compose

  ## Project Structure

  ```text
  .
  ├── docker-compose.yml
  ├── requirements.txt
  ├── .env.example
  ├── src/
  │   ├── ingest.py
  │   ├── search.py
  │   └── chat.py
  ├── document.pdf
  └── README.md

  ## Prerequisites

  - Python 3.10+
  - Docker and Docker Compose
  - Git

  ## Environment Setup

  1. Clone the repository:

  git clone https://github.com/luizpenna/langchain-pgvector-semantic-search.git
  cd langchain-pgvector-semantic-search

  2. Create and activate a virtual environment:

  python3 -m venv venv
  source venv/bin/activate

  3. Install dependencies:

  pip install -r requirements.txt

  4. Create your environment file:

  cp .env.example .env

  5. Edit .env and add your API keys:

  - OPENAI_API_KEY=
  - GOOGLE_API_KEY= 

  ## Environment Variables (.env)

  Example:

  GOOGLE_API_KEY=
  GOOGLE_EMBEDDING_MODEL=models/embedding-001

  OPENAI_API_KEY=
  OPENAI_EMBEDDING_MODEL=text-embedding-3-small

  DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
  PG_VECTOR_COLLECTION_NAME=document_chunks
  PDF_PATH=document.pdf

  ## Run Order (Required)

  1. Start PostgreSQL + pgVector:

  docker compose up -d

  2. Run PDF ingestion:

  python src/ingest.py

  3. Start the CLI chat:

  python src/chat.py

  ## Chat Response Rules

  The system must answer only using information from retrieved PDF context.

  If the answer is not explicitly present in the context, it must return exactly:

  I do not have enough information to answer your question.

  ## CLI Example

  QUESTION: What is SuperTechIABrazil's revenue?
  ANSWER: The revenue was 10 million BRL.

  QUESTION: How many customers do we have in 2024?
  ANSWER: I do not have enough information to answer your question.

  ## Troubleshooting

  1. Database connection error:

  - Check running containers:

  docker compose ps

  2. Missing vector extension:

  - Verify installed extensions:

  docker compose exec postgres psql -U postgres -d rag -c "\dx"

  3. API key error:

  - Confirm .env is filled correctly and loaded.
