import os
from pathlib import Path

import ollama


def main() -> None:
    if not hasattr(ollama, "chat"):
        module_file = getattr(ollama, "__file__", "<unknown>")
        raise RuntimeError(
            f"Imported wrong 'ollama' module from {module_file}. "
            "Ensure the official ollama package is installed and not shadowed by local files."
        )

    system_prompt_path = Path(__file__).with_name("system_prompt.txt")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip() or "qwen2.5:3b"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Hi"},
        ],
    )

    print(response.get("message", {}).get("content", "").strip())


if __name__ == "__main__":
    main()
