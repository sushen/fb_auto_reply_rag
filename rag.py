"""
RAG Engine - Core retrieval and question answering system.
"""

import os
import sqlite3
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Database for user memories
MEMORY_DB = 'user_memories.db'

ConversationPair = dict[str, str]
ConversationDataset = dict[str, list[ConversationPair]]
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
DEFAULT_CONTEXT_EXAMPLES = 4
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


def get_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    return conn


def init_memory_db():
    conn = get_memory_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_memories (
            user_id TEXT PRIMARY KEY,
            history TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def save_user_memory(user_id, history):
    conn = get_memory_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO user_memories (user_id, history, updated_at)
        VALUES (?, ?, ?)
    ''', (user_id, json.dumps(history), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def load_user_memory(user_id):
    if not user_id:
        return []
    conn = get_memory_db()
    cursor = conn.cursor()
    cursor.execute('SELECT history FROM user_memories WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return json.loads(result[0])
    return []


def load_conversation_dataset(path: Path) -> ConversationDataset:
    """
    Load conversation guidance data from JSON.

    Supported formats:
    - Stage map: {"stage_name": [{"question": "...", "answer": "..."}]}
    - Conversation list: [{"messages": [{"role": "user|assistant", "content": "..."}]}]
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
        raise ValueError("Unsupported conversation dataset format.")

    if not dataset:
        raise ValueError("No usable question/answer pairs found in conversation dataset.")
    return dataset


def _is_stage_dataset(dataset: ConversationDataset) -> bool:
    if not dataset:
        return False
    if "stage_imported_conversations" in dataset:
        return len(dataset) > 1
    return True


def _extract_pairs_from_entries(entries: list[Any]) -> list[ConversationPair]:
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
    """Normalize text for robust string matching."""
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def _score_example(user_message: str, example_question: str) -> float:
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


def _guess_stage_from_keywords(user_message: str, dataset: ConversationDataset) -> str:
    """Infer likely stage from keyword hints for very short/ambiguous user messages."""
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


def build_context_block(
    user_message: str,
    dataset: ConversationDataset,
    max_examples: int = DEFAULT_CONTEXT_EXAMPLES,
) -> str:
    """Build a stage-aware conversation examples block from relevant dataset rows."""
    top_stage, selected_examples = _select_examples_for_best_stage(user_message, dataset, max_examples)
    if not selected_examples:
        return ""

    lines = [
        "Conversation stage examples:",
        f"Likely stage: {top_stage}",
    ]
    for _score, _index, stage_name, example in selected_examples:
        lines.append("")
        lines.append(f"Stage: {stage_name}")
        lines.append(f"User: {example['question']}")
        lines.append(f"Assistant: {example['answer']}")
    return "\n".join(lines)


class RAGSystem:
    def __init__(self, upload_folder='uploads', llm_model="qwen2.5:3b", embed_model="bge-m3:latest"):
        self.upload_folder = upload_folder
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.vector_store = None
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)
        self.guidance_system_prompt = ""
        self.guidance_dataset: ConversationDataset = {}
        logger.info(f"Initializing RAG system with {llm_model}...")
        self._load_conversation_guidance()
        self.load_documents()
        self.initialize_chain()
    
    def reload(self):
        logger.info("Reloading knowledge base...")
        self._load_conversation_guidance()
        self.load_documents()
        self.initialize_chain()
        logger.info("Knowledge base reloaded")
    
    def query(self, message, user_id=None):
        """
        Query the RAG system.
        
        Args:
            message: User message
            user_id: If provided, loads/persists conversation memory in SQLite.
                     If None, uses fresh memory (no persistence).
        """
        clean_message = str(message or "").strip()
        if not clean_message:
            return {"error": "No message provided"}, 400

        # Load user-specific memory from database (only if user_id provided)
        history = load_user_memory(user_id) if user_id else []

        direct_guidance_answer = self._get_direct_guidance_answer(clean_message)
        if direct_guidance_answer:
            if user_id:
                updated_history = [
                    {"type": "human", "content": clean_message},
                    {"type": "ai", "content": direct_guidance_answer},
                ]
                save_user_memory(user_id, history + updated_history)
            logger.info(
                "Guidance direct answer used for user %s: %s...",
                user_id or "web",
                clean_message[:30],
            )
            return {"response": direct_guidance_answer}

        # Fallback to normal chat if no RAG documents are available.
        if not self.vector_store:
            try:
                prompt_parts = [
                    "You are a helpful, natural conversational assistant.",
                    "Respond like a human assistant in clear, friendly English.",
                ]
                if self.guidance_system_prompt:
                    prompt_parts.append("Follow these conversation rules:")
                    prompt_parts.append(self.guidance_system_prompt)

                context_block = self._build_guidance_context(clean_message)
                if context_block:
                    prompt_parts.append(context_block)

                for msg in history[-10:]:
                    if msg.get('type') == 'human':
                        prompt_parts.append(f"User: {msg.get('content', '')}")
                    elif msg.get('type') == 'ai':
                        prompt_parts.append(f"Assistant: {msg.get('content', '')}")

                prompt_parts.append(f"User: {clean_message}")
                prompt_parts.append("Assistant:")
                prompt = "\n".join(prompt_parts)

                answer = str(self.llm.invoke(prompt)).strip()
                if not answer:
                    answer = "I am here. Tell me what you want to talk about."

                if user_id:
                    updated_history = [
                        {"type": "human", "content": clean_message},
                        {"type": "ai", "content": answer}
                    ]
                    save_user_memory(user_id, history + updated_history)

                logger.info(f"Chat fallback used for user {user_id or 'web'}: {clean_message[:30]}...")
                return {"response": answer}
            except Exception as e:
                logger.error(f"Fallback chat error: {e}")
                return {"response": f"Error processing your query: {str(e)}"}, 500

        try:
            # Create memory with history
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
            
            # Load history into memory
            for msg in history:
                if msg['type'] == 'human':
                    memory.chat_memory.add_user_message(msg['content'])
                elif msg['type'] == 'ai':
                    memory.chat_memory.add_ai_message(msg['content'])
            
            # Create chain with user memory
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                memory=memory,
                return_source_documents=True,
                verbose=True
            )
            
            # Query
            guided_message = self._augment_message_with_guidance(clean_message)
            result = qa_chain.invoke(guided_message)
            
            # Save updated memory to database (only if user_id provided)
            if user_id:
                updated_history = [
                    {"type": "human", "content": clean_message},
                    {"type": "ai", "content": result.get('answer', str(result))}
                ]
                save_user_memory(user_id, history + updated_history)
            
            logger.info(f"Query processed for user {user_id or 'web'}: {clean_message[:30]}...")
            return {"response": result.get('answer', str(result))}
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"response": f"Error processing your query: {str(e)}"}, 500
    
    def load_documents(self):
        # Clear existing vector store first
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except:
                pass
            self.vector_store = None
        
        if not os.path.exists(self.upload_folder):
            return []
        
        documents = []
        for root, dirs, files in os.walk(self.upload_folder):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                try:
                    if filename.endswith('.txt'):
                        loader = TextLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {filename}")
                    elif filename.endswith('.pdf'):
                        loader = PyPDFLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {filename}")
                    elif filename.endswith('.docx'):
                        loader = Docx2txtLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {filename}")
                    elif filename.endswith('.csv'):
                        loader = CSVLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            docs = text_splitter.split_documents(documents)
            self.vector_store = Chroma.from_documents(docs, self.embeddings)
        
        return documents
    
    def initialize_chain(self):
        # Chain is created per-query with user memory
        pass

    def _load_conversation_guidance(self):
        """Load system prompt and conversation examples used to guide responses."""
        base_dir = Path(__file__).resolve().parent
        system_prompt_path = base_dir / "ollama_train" / "system_prompt.txt"
        dataset_path = base_dir / "ollama_train" / "dataset" / "context" / "conversation_stages.json"
        backup_dataset_path = base_dir / "ollama_train" / "dataset" / "context" / "conversation_stages_backups.json"

        self.guidance_system_prompt = ""
        self.guidance_dataset = {}

        try:
            if system_prompt_path.exists():
                self.guidance_system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning(f"Failed to load system prompt from {system_prompt_path}: {exc}")

        try:
            if dataset_path.exists():
                loaded_dataset = load_conversation_dataset(dataset_path)
                if _is_stage_dataset(loaded_dataset):
                    self.guidance_dataset = loaded_dataset
                elif backup_dataset_path.exists():
                    backup_dataset = load_conversation_dataset(backup_dataset_path)
                    if _is_stage_dataset(backup_dataset):
                        self.guidance_dataset = backup_dataset
                        logger.warning(
                            "conversation_stages.json is not stage-based; using backup stage dataset: %s",
                            backup_dataset_path,
                        )
                    else:
                        self.guidance_dataset = loaded_dataset
                else:
                    self.guidance_dataset = loaded_dataset
        except Exception as exc:
            logger.warning(f"Failed to load conversation dataset from {dataset_path}: {exc}")

        dataset_pairs = sum(len(v) for v in self.guidance_dataset.values())
        logger.info(
            "Conversation guidance loaded: system_prompt=%s, stages=%s, pairs=%s",
            bool(self.guidance_system_prompt),
            len(self.guidance_dataset),
            dataset_pairs,
        )

    def _build_guidance_context(self, user_message: str) -> str:
        if not self.guidance_dataset:
            return ""

        max_examples_raw = os.getenv("OLLAMA_CONTEXT_EXAMPLES", str(DEFAULT_CONTEXT_EXAMPLES)).strip()
        try:
            max_examples = max(1, min(int(max_examples_raw), 12))
        except ValueError:
            max_examples = DEFAULT_CONTEXT_EXAMPLES

        return build_context_block(user_message, self.guidance_dataset, max_examples=max_examples)

    def _augment_message_with_guidance(self, user_message: str) -> str:
        """
        Prefix user input with system rules and relevant conversation examples.

        This mirrors the context-injection behavior used by ollama_train/ollama_train.py.
        """
        context_block = self._build_guidance_context(user_message)
        if not self.guidance_system_prompt and not context_block:
            return user_message

        parts = []
        if self.guidance_system_prompt:
            parts.append("Conversation rules:")
            parts.append(self.guidance_system_prompt)
        if context_block:
            parts.append(context_block)
            parts.append("Follow the stage flow shown above when replying.")
        parts.append(f"User: {user_message}")
        return "\n\n".join(parts)

    def _get_direct_guidance_answer(self, user_message: str) -> str:
        """
        Return a deterministic stage answer for strong matches.

        This prevents drift to generic replies when the dataset already has a clear answer.
        """
        if not self.guidance_dataset:
            return ""

        normalized_user = _normalize_text(user_message)
        ranked = _rank_examples(user_message, self.guidance_dataset)
        if ranked:
            best_score, _index, _stage, best_example = ranked[0]
            if _normalize_text(best_example["question"]) == normalized_user:
                return best_example["answer"]

            # If overlap is strong (multi-token), trust dataset answer directly.
            token_count = len(_tokenize(user_message))
            if token_count >= 2 and best_score >= 6.0:
                return best_example["answer"]

        # For short prompts like "crypto", map to a stage hint.
        if len(_tokenize(user_message)) <= 3:
            hinted_stage = _guess_stage_from_keywords(user_message, self.guidance_dataset)
            if hinted_stage and self.guidance_dataset.get(hinted_stage):
                return self.guidance_dataset[hinted_stage][0]["answer"]

        return ""
