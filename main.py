"""
main.py
-------
CLI entry point for IntelliMoE — Multi-Expert AI Assistant.

Usage:
    python main.py
    python main.py --query "How do I implement a binary search tree?"
    python main.py --query "Build an AI Hospital Management System" --verbose

For the Streamlit UI, run:
    streamlit run ui/app.py
"""

import argparse
import logging
import sys

from utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="intellimoe",
        description="IntelliMoE — Multi-Expert AI Assistant powered by TinyLlama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --query 'What is backpropagation?'\n"
            "  python main.py --query 'Design a URL shortener' --verbose\n"
            "\nFor the web UI, run:\n"
            "  streamlit run ui/app.py"
        ),
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="The question to ask. If omitted, enters interactive REPL mode.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def run_query(router, query: str) -> None:
    """Route a single query and print the result."""
    logger = logging.getLogger(__name__)

    # Show which experts will be activated (no inference yet).
    selected = router.selected_experts(query)
    expert_labels = ", ".join(
        f"{e.value.replace('_', ' ').title()}" for e in selected
    )
    print(f"\n{'─' * 60}")
    print(f"🔍 Query     : {query}")
    print(f"🎯 Expert(s) : {expert_labels}")
    print(f"{'─' * 60}")

    try:
        answer = router.route(query)
        print(f"\n{answer}\n")
    except NotImplementedError as exc:
        logger.error("Expert not yet available: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Inference error: %s", exc)
        sys.exit(1)


def run_repl(router) -> None:
    """Interactive read-eval-print loop."""
    print("\n🧠 IntelliMoE — Interactive Mode")
    print("   Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye! 👋")
            break

        run_query(router, query)


def main() -> None:
    args = parse_args()

    # Initialise logging before any other imports that use it.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    logger = logging.getLogger(__name__)
    logger.info("IntelliMoE starting …")

    # Lazy-import router after logging is configured.
    from router.router import ExpertRouter  # noqa: PLC0415

    router = ExpertRouter()

    if args.query:
        run_query(router, args.query)
    else:
        run_repl(router)


if __name__ == "__main__":
    main()
