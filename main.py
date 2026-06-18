"""Interface CLI pour l'agent RAG agentique."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import ask


def main():
    print("=" * 60)
    print("  Agentic RAG — Intelligence Artificielle")
    print("  Tapez 'quit' pour quitter")
    print("=" * 60)

    thread_id = "cli-session"

    while True:
        try:
            question = input("\nQuestion : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir.")
            break

        print("\nRéflexion en cours...\n")
        result = ask(question, thread_id=thread_id)
        print(result.get("generation", "Pas de réponse."))


if __name__ == "__main__":
    main()
