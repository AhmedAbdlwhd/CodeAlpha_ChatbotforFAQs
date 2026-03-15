"""
main.py — Command-line entry point for the University FAQ Chatbot

Usage:
    .venv\\Scripts\\python.exe main.py

For the Streamlit interface run:
    .venv\\Scripts\\streamlit.exe run ui/streamlit_ui.py
"""

import os
import sys

# Ensure the project root is always on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot.faq_engine import FAQEngine


def main() -> None:
    print("=" * 60)
    print("  University FAQ Chatbot")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    engine = FAQEngine(threshold=0.4)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "bye"}:
            print("Chatbot: Goodbye! Have a great day.")
            break

        result = engine.ask(question)

        print(f"\nChatbot: {result['answer']}")
        print(
            f"  [score: {result['score']:.4f} | "
            f"matched: \"{result['matched_question']}\"]"
        )

        if not result["confident"]:
            print("  [This question has been logged for review.]")


if __name__ == "__main__":
    main()
