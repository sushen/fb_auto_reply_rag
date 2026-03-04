import json
import os
import re
from pathlib import Path
from typing import Any

import ollama


ConversationPair = dict[str, str]
ConversationDataset = dict[str, list[ConversationPair]]

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_MAX_EXAMPLES = 4
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def load_system_prompt(path: Path) -> str:
    """Load the system prompt that controls assistant behavior."""
    return path.read_text(encoding="utf-8").strip()


def load_conversation_dataset(path: Path) -> ConversationDataset:
    """
    Load the conversation dataset from JSON.

    Supported top-level formats:
    1) Stage map:
       {
         "stage_name": [{"question": "...", "answer": "..."}]
       }
    2) Conversation list:
       [
         {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
       ]

    Invalid entries are skipped so the loader remains resilient as the dataset evolves.
    """
    raw_payload = json.loads(path.read_text(encoding="utf-8"))

    dataset: ConversationDataset = {}

    if isinstance(raw_payload, dict):
        for stage_name, entries in raw_payload.items():
            if not isinstance(stage_name, str) or not isinstance(entries, list):
                continue

            normalized_entries = _extract_pairs_from_entries(entries)
            if normalized_entries:
                dataset[stage_name] = normalized_entries
    elif isinstance(raw_payload, list):
        normalized_entries = _extract_pairs_from_entries(raw_payload)
        if normalized_entries:
            dataset["stage_imported_conversations"] = normalized_entries
    else:
        raise ValueError(
            "Unsupported dataset format. Expected a JSON object (stage map) or a JSON array (conversation list)."
        )

    if not dataset:
        raise ValueError(
            f"No usable question/answer pairs found in dataset: {path}"
        )

    return dataset


def _extract_pairs_from_entries(entries: list[Any]) -> list[ConversationPair]:
    """Normalize mixed dataset entries into question/answer pairs."""
    normalized_entries: list[ConversationPair] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        question = str(entry.get("question", "")).strip()
        answer = str(entry.get("answer", "")).strip()
        if question and answer:
            normalized_entries.append({"question": question, "answer": answer})
            continue

        messages = entry.get("messages")
        normalized_entries.extend(_extract_pairs_from_messages(messages))
    return normalized_entries


def _extract_pairs_from_messages(messages: Any) -> list[ConversationPair]:
    """Extract sequential user->assistant pairs from a message list."""
    if not isinstance(messages, list):
        return []

    pairs: list[ConversationPair] = []
    pending_user: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            pending_user = content
            continue

        if role == "assistant" and pending_user:
            pairs.append({"question": pending_user, "answer": content})
            pending_user = None

    return pairs


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def _score_example(user_message: str, example_question: str) -> float:
    """
    Lightweight lexical relevance score.

    Higher score means the example question is more semantically aligned with user input.
    """
    user_lower = user_message.lower().strip()
    question_lower = example_question.lower().strip()
    if not user_lower or not question_lower:
        return 0.0

    if user_lower == question_lower:
        return 100.0
    if question_lower in user_lower or user_lower in question_lower:
        return 10.0

    user_tokens = _tokenize(user_lower)
    question_tokens = _tokenize(question_lower)
    if not user_tokens or not question_tokens:
        return 0.0

    intersection = len(user_tokens & question_tokens)
    if intersection == 0:
        return 0.0

    # Favors overlap while still rewarding shorter, tighter example matches.
    return intersection + (intersection / len(question_tokens))


def select_relevant_examples(
    user_message: str,
    dataset: ConversationDataset,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> list[ConversationPair]:
    """
    Select the most relevant examples across all stages.

    Falls back to the earliest dataset entries if no lexical match is found.
    """
    scored_examples: list[tuple[float, int, ConversationPair]] = []
    index = 0
    for stage_entries in dataset.values():
        for example in stage_entries:
            score = _score_example(user_message, example["question"])
            scored_examples.append((score, index, example))
            index += 1

    if not scored_examples:
        return []

    positive_matches = [item for item in scored_examples if item[0] > 0]
    source = positive_matches if positive_matches else scored_examples
    source.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in source[:max_examples]]


def build_context_block(
    user_message: str,
    dataset: ConversationDataset,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> str:
    """
    Build a structured context section from relevant conversation examples.

    This block is prepended to the live user input before calling Ollama.
    """
    examples = select_relevant_examples(user_message=user_message, dataset=dataset, max_examples=max_examples)
    if not examples:
        return "Conversation examples:\nNo examples available."

    lines = ["Conversation examples:"]
    for example in examples:
        lines.append("")
        lines.append(f"User: {example['question']}")
        lines.append(f"Assistant: {example['answer']}")
    return "\n".join(lines)


def _extract_assistant_content(response: Any) -> str:
    """Support dict-like and object-like responses returned by different ollama client versions."""
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", "")).strip()

    message = getattr(response, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    return str(content).strip()


def chat_with_context(
    user_input: str,
    *,
    model: str,
    system_prompt: str,
    dataset: ConversationDataset,
) -> str:
    """Send a single user turn to Ollama with system prompt + dataset-derived context."""
    context = build_context_block(user_input, dataset)
    user_message = f"{context}\n\nUser: {user_input}"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return _extract_assistant_content(response)


def _validate_ollama_import() -> None:
    if hasattr(ollama, "chat"):
        return

    module_file = getattr(ollama, "__file__", "<unknown>")
    raise RuntimeError(
        f"Imported wrong 'ollama' module from {module_file}. "
        "Ensure the official ollama package is installed and not shadowed by local files."
    )


def run_terminal_loop() -> None:
    """Interactive loop for `python -m ollama_train.ollama_train`."""
    base_dir = Path(__file__).resolve().parent
    system_prompt_path = base_dir / "system_prompt.txt"
    dataset_path = base_dir / "dataset" / "context" / "conversation_stages.json"

    system_prompt = load_system_prompt(system_prompt_path)
    dataset = load_conversation_dataset(dataset_path)
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    print(f"[ollama_train] model: {model}")
    print("[ollama_train] type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        try:
            assistant_reply = chat_with_context(
                user_input=user_input,
                model=model,
                system_prompt=system_prompt,
                dataset=dataset,
            )
        except Exception as exc:
            print(f"Assistant: Error calling Ollama: {exc}")
            continue

        if not assistant_reply:
            assistant_reply = "I could not generate a response."
        print(f"Assistant: {assistant_reply}")


def main() -> None:
    _validate_ollama_import()
    run_terminal_loop()


if __name__ == "__main__":
    main()
