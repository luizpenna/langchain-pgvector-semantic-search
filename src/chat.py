import argparse
import os
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from search import search

# Prompt contract required by the project:
# - The model must answer only from retrieved context.
# - If context does not explicitly contain the answer, it must return the fallback sentence.
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def parse_args() -> argparse.Namespace:
    # CLI flags let you choose providers/models at runtime without code changes.
    parser = argparse.ArgumentParser(description="CLI de busca semantica com RAG.")
    parser.add_argument(
        "--provider",
        choices=["openai", "google"],
        help="Provider de embedding (usado no search.py).",
    )
    parser.add_argument(
        "--collection-name",
        dest="collection_name",
        help="Nome da coleção vetorial.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "google"],
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="Provider da LLM para resposta.",
    )
    parser.add_argument(
        "--model",
        help="Modelo da LLM. Se omitido, usa o default por provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperatura da LLM.",
    )
    parser.add_argument("--logs", action="store_true", help="Ativa logs de debug.")
    return parser.parse_args()


def format_context(results: list[dict[str, object]]) -> str:
    # Convert retrieved chunks into a single text block for prompt injection.
    # search.py already returns at most k=10 results.
    if not results:
        # Empty context forces the LLM to follow the fallback rule in the prompt.
        return ""
    return "\n\n".join(str(item.get("page_content", "")).strip() for item in results)


def build_llm(args: argparse.Namespace):
    # Build the chat model according to selected provider.
    # This controls only answer generation; embeddings are controlled in search.py.
    llm_provider = (args.llm_provider or "openai").strip().lower()

    if llm_provider == "google":
        model = args.model or os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=args.temperature), model

    model = args.model or os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano")
    return ChatOpenAI(model=model, temperature=args.temperature), model


def answer_question(
    pergunta: str,
    search_args: SimpleNamespace,
    llm,
) -> str:
    # Retrieval step (RAG):
    # 1) Use the user question to query pgvector.
    # 2) Merge retrieved chunks into CONTEXTO.
    # 3) Ask the LLM with strict grounding rules.
    results = search(pergunta, search_args)
    contexto = format_context(results)

    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
    response = llm.invoke(prompt)
    return str(response.content).strip()


def main() -> None:
    # Load environment variables from .env (API keys, model defaults, DB config).
    load_dotenv()
    args = parse_args()

    # search.py expects an args-like object with provider/collection/logs fields.
    search_args = SimpleNamespace(
        provider=args.provider,
        collection_name=args.collection_name,
        logs=args.logs,
    )

    # Instantiate the answer LLM once and reuse it for all turns.
    llm, model = build_llm(args)

    if args.logs:
        print(f"[debug] Embedding provider: {args.provider or os.getenv('EMBEDDING_PROVIDER', 'openai')}")
        print(f"[debug] LLM provider: {args.llm_provider}")
        print(f"[debug] LLM model: {model}")

    # Interactive chat loop for terminal usage.
    # Type "sair", "exit", or "quit" to finish.
    print("Faça sua pergunta (digite 'sair' para encerrar):")
    while True:
        pergunta = input("\nPERGUNTA: ").strip()
        if not pergunta:
            # Ignore empty inputs and keep waiting for a valid question.
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            break

        # Generate grounded answer from retrieved PDF context.
        resposta = answer_question(pergunta, search_args, llm)
        print(f"RESPOSTA: {resposta}")


if __name__ == "__main__":
    main()
