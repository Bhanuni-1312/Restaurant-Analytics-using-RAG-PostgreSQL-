from dotenv import load_dotenv
load_dotenv()

from app.rag import SchemaRAG


def main():
    rag = SchemaRAG()

    # Index schema once (safe to run multiple times)
    rag.index_schema()

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        try:
            answer = rag.answer_question(q)
            # IMPORTANT: print ONLY final answer (your requirement)
            print("\nAnswer:\n", answer)
        except Exception as e:
            print("\nError:", str(e))


if __name__ == "__main__":
    main()
