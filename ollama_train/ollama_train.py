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
STAGE_KEYWORD_HINTS: dict[str, set[str]] = {
    "stage_2_identify_interest": {"trade", "trading", "earn", "earning", "learn", "learning"},
    "stage_3_trading_ai_bot": {"crypto", "bot", "binance", "algorithmic", "ai"},
    "stage_4_learning_program": {"learn", "course", "training", "python", "study"},
    "stage_5_passive_investment": {"invest", "investment", "portfolio", "passive"},
    "stage_6_affiliate_program": {"affiliate", "refer", "referral", "promote", "commission"},
    "stage_7_binance_setup": {"start", "setup", "register", "binance", "account"},
    "stage_8_license_and_community": {"discord", "license", "community", "support"},
    "stage_9_schedule_call": {"call", "appointment", "meeting"},
}


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


def _is_stage_dataset(dataset: ConversationDataset) -> bool:
    if not dataset:
        return False
    if "stage_imported_conversations" in dataset:
        return len(dataset) > 1
    return True


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


def _normalize_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def _score_example(user_message: str, example_question: str) -> float:
    """
    Lightweight lexical relevance score.

    Higher score means the example question is more semantically aligned with user input.
    """
    user_tokens = _tokenize(user_message)
    question_tokens = _tokenize(example_question)
    if not user_tokens or not question_tokens:
        return 0.0

    if _normalize_text(user_message) == _normalize_text(example_question):
        return 100.0

    intersection = len(user_tokens & question_tokens)
    if intersection == 0:
        return 0.0

    question_coverage = intersection / len(question_tokens)
    user_coverage = intersection / len(user_tokens)
    return intersection + question_coverage + user_coverage


def _guess_stage_from_keywords(user_message: str, dataset: ConversationDataset) -> str:
    user_tokens = _tokenize(user_message)
    if not user_tokens:
        return ""

    best_stage = ""
    best_score = 0
    for stage_name, keywords in STAGE_KEYWORD_HINTS.items():
        if stage_name not in dataset:
            continue
        overlap = len(user_tokens & keywords)
        if overlap > best_score:
            best_score = overlap
            best_stage = stage_name
    return best_stage if best_score > 0 else ""


def _rank_examples(
    user_message: str,
    dataset: ConversationDataset,
) -> list[tuple[float, int, str, ConversationPair]]:
    scored_examples: list[tuple[float, int, str, ConversationPair]] = []
    index = 0
    for stage_name, stage_entries in dataset.items():
        for example in stage_entries:
            score = _score_example(user_message, example["question"])
            scored_examples.append((score, index, stage_name, example))
            index += 1
    scored_examples.sort(key=lambda item: (-item[0], item[1]))
    return scored_examples


def _select_examples_for_best_stage(
    user_message: str,
    dataset: ConversationDataset,
    max_examples: int,
) -> tuple[str, list[tuple[float, int, str, ConversationPair]]]:
    ranked = _rank_examples(user_message, dataset)
    if not ranked:
        return "", []

    top_stage = ranked[0][2]
    if ranked[0][0] <= 0:
        hinted_stage = _guess_stage_from_keywords(user_message, dataset)
        if hinted_stage:
            top_stage = hinted_stage

    stage_examples = [item for item in ranked if item[2] == top_stage]
    positive_stage_examples = [item for item in stage_examples if item[0] > 0]
    selected = positive_stage_examples if positive_stage_examples else stage_examples
    return top_stage, selected[:max_examples]


def _get_direct_guidance_answer(user_message: str, dataset: ConversationDataset) -> str:
    if not dataset:
        return ""

    normalized_user = _normalize_text(user_message)
    ranked = _rank_examples(user_message, dataset)
    if ranked:
        best_score, _index, _stage, best_example = ranked[0]
        if _normalize_text(best_example["question"]) == normalized_user:
            return best_example["answer"]

        token_count = len(_tokenize(user_message))
        if token_count >= 2 and best_score >= 6.0:
            return best_example["answer"]

    if len(_tokenize(user_message)) <= 3:
        hinted_stage = _guess_stage_from_keywords(user_message, dataset)
        if hinted_stage and dataset.get(hinted_stage):
            return dataset[hinted_stage][0]["answer"]

    return ""


def select_relevant_examples(
    user_message: str,
    dataset: ConversationDataset,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> list[ConversationPair]:
    """
    Select examples from the most relevant stage only to keep context coherent.
    """
    _top_stage, selected = _select_examples_for_best_stage(user_message, dataset, max_examples)
    return [item[3] for item in selected]


def build_context_block(
    user_message: str,
    dataset: ConversationDataset,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> str:
    """
    Build a structured context section from relevant conversation examples.

    This block is prepended to the live user input before calling Ollama.
    """
    top_stage, selected = _select_examples_for_best_stage(user_message, dataset, max_examples)
    if not selected:
        return "Conversation examples:\nNo examples available."

    lines = [
        "Conversation stage examples:",
        f"Likely stage: {top_stage}",
    ]
    for _score, _index, stage_name, example in selected:
        lines.append("")
        lines.append(f"Stage: {stage_name}")
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
    direct_answer = _get_direct_guidance_answer(user_input, dataset)
    if direct_answer:
        return direct_answer

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
    backup_dataset_path = base_dir / "dataset" / "context" / "conversation_stages_backups.json"

    system_prompt = load_system_prompt(system_prompt_path)
    dataset = load_conversation_dataset(dataset_path)
    if not _is_stage_dataset(dataset) and backup_dataset_path.exists():
        backup_dataset = load_conversation_dataset(backup_dataset_path)
        if _is_stage_dataset(backup_dataset):
            dataset = backup_dataset
            print(f"[ollama_train] using backup stage dataset: {backup_dataset_path}")
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
