#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError
except Exception:  # pragma: no cover
    snapshot_download = None
    LocalEntryNotFoundError = Exception

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def _configure_storage(storage_dir: Optional[str], timezone_name: Optional[str]) -> None:
    default_storage = ROOT / "data"
    os.environ.setdefault("ACE_STORAGE_DIR", str(default_storage))
    if storage_dir:
        os.environ["ACE_STORAGE_DIR"] = storage_dir
    if timezone_name:
        os.environ["ACE_USER_TZ"] = timezone_name


DEFAULT_MODEL = os.getenv("ACE_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_DEVICE = os.getenv("ACE_DEVICE", "auto")
DEFAULT_CHAT_ID = os.getenv("ACE_CHAT_ID", "default")
DEFAULT_TOP_K = int(os.getenv("ACE_TOP_K", "5"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("ACE_MAX_NEW_TOKENS", "400"))
DEFAULT_EXTRACT_MAX_NEW_TOKENS = int(os.getenv("ACE_EXTRACT_MAX_NEW_TOKENS", "200"))
DEFAULT_TEMPERATURE = float(os.getenv("ACE_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("ACE_TOP_P", "0.9"))
DEFAULT_EVENT_CONTEXT_BUDGET_TOKENS = int(os.getenv("ACE_EVENT_CONTEXT_BUDGET_TOKENS", "1200"))
DEFAULT_REFRESH_EVERY = int(os.getenv("ACE_REFRESH_EVERY", "3"))
DEFAULT_EPISODE_LOOKBACK_MONTHS = int(os.getenv("ACE_EPISODE_LOOKBACK_MONTHS", "12"))
DEFAULT_EPISODE_EVENTS_PER_CONTEXT = int(os.getenv("ACE_EPISODE_EVENTS_PER_CONTEXT", "5"))
DEFAULT_EPISODE_TOP_K = int(os.getenv("ACE_EPISODE_TOP_K", "5"))
DEFAULT_EPISODE_PREFILTER = os.getenv("ACE_EPISODE_PREFILTER", "1").lower() in {"1", "true", "yes"}
DEFAULT_MEMORY_SCORE_THRESHOLD = float(os.getenv("ACE_MEMORY_SCORE_THRESHOLD", "0.35"))
DEFAULT_ALLOW_GENERAL_FALLBACK = os.getenv("ACE_ALLOW_GENERAL_FALLBACK", "1").lower() in {"1", "true", "yes"}
DEFAULT_MODEL_DTYPE = os.getenv("ACE_MODEL_DTYPE", "float16")
DEFAULT_LOW_CPU_MEM = os.getenv("ACE_LOW_CPU_MEM", "1").lower() in {"1", "true", "yes"}
DEFAULT_FORCE_DOWNLOAD = os.getenv("ACE_FORCE_DOWNLOAD", "0").lower() in {"1", "true", "yes"}
DEFAULT_HF_CACHE_DIR = os.getenv("ACE_HF_CACHE_DIR")
DEFAULT_STREAM = os.getenv("ACE_STREAM", "1").lower() in {"1", "true", "yes"}
DEFAULT_DATASET_ID = os.getenv("ACE_HF_DATASET", "")
DEFAULT_DATASET_CONFIG = os.getenv("ACE_HF_DATASET_CONFIG", "events")
DEFAULT_EMBED_MODEL_NAME = os.getenv("ACE_EMBED_MODEL_NAME", "intfloat/e5-large-v2")

parser = argparse.ArgumentParser(description="Run the ACE minimal RAG agent.")
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu, cuda, mps, or auto")
parser.add_argument("--storage-dir", default=None, help="Override storage directory")
parser.add_argument("--timezone", default=None, help="IANA timezone name, e.g. America/New_York")
parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
parser.add_argument(
    "--extract-max-new-tokens",
    type=int,
    default=DEFAULT_EXTRACT_MAX_NEW_TOKENS,
    help="Max tokens for action/evidence extraction JSON.",
)
parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
parser.add_argument("--event-context-budget", type=int, default=DEFAULT_EVENT_CONTEXT_BUDGET_TOKENS)
parser.add_argument("--chat-id", default=DEFAULT_CHAT_ID)
parser.add_argument("--episode-lookback-months", type=int, default=DEFAULT_EPISODE_LOOKBACK_MONTHS)
parser.add_argument("--episode-events-per-context", type=int, default=DEFAULT_EPISODE_EVENTS_PER_CONTEXT)
parser.add_argument("--episode-top-k", type=int, default=DEFAULT_EPISODE_TOP_K)
parser.add_argument(
    "--episode-prefilter",
    action=argparse.BooleanOptionalAction,
    default=DEFAULT_EPISODE_PREFILTER,
)
parser.add_argument(
    "--refresh-every",
    type=int,
    default=DEFAULT_REFRESH_EVERY,
    help="Rebuild retrieval index every N queries (0 to disable).",
)
parser.add_argument(
    "--memory-score-threshold",
    type=float,
    default=DEFAULT_MEMORY_SCORE_THRESHOLD,
)
parser.add_argument(
    "--allow-general-fallback",
    action=argparse.BooleanOptionalAction,
    default=DEFAULT_ALLOW_GENERAL_FALLBACK,
)
parser.add_argument(
    "--model-dtype",
    default=DEFAULT_MODEL_DTYPE,
    choices=["float16", "bfloat16", "float32"],
)
parser.add_argument(
    "--low-cpu-mem",
    action=argparse.BooleanOptionalAction,
    default=DEFAULT_LOW_CPU_MEM,
)
parser.add_argument(
    "--force-download",
    action=argparse.BooleanOptionalAction,
    default=DEFAULT_FORCE_DOWNLOAD,
    help="Force re-download of model/tokenizer files even if cached.",
)
parser.add_argument(
    "--hf-cache-dir",
    default=DEFAULT_HF_CACHE_DIR,
    help="Override Hugging Face cache directory for model files.",
)
parser.add_argument(
    "--stream",
    action=argparse.BooleanOptionalAction,
    default=DEFAULT_STREAM,
    help="Stream tokens as they are generated.",
)
parser.add_argument(
    "--use-ace-dataset",
    action="store_true",
    default=False,
    help="Download the ACE dataset, build episodes/embeddings, and include it in retrieval.",
)
parser.add_argument(
    "--ace-dataset",
    default=DEFAULT_DATASET_ID,
    help="Hugging Face dataset ID to use when --use-ace-dataset is set.",
)
parser.add_argument(
    "--ace-dataset-config",
    default=DEFAULT_DATASET_CONFIG,
    help="Dataset config name (defaults to 'events').",
)
parser.add_argument(
    "--rebuild-dataset-embeddings",
    action="store_true",
    default=False,
    help="Force rebuild of dataset event embeddings.",
)
args = parser.parse_args()

_configure_storage(args.storage_dir, args.timezone)

import storage_helpers as storage
from ace_agent import run_chat_loop


MEMORY_SYSTEM_PROMPT = (
    "You are a personal assistant with access to prior interactions. "
    "Answer only using the provided context; if the answer is not in memory, say so."
)
GENERAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use general knowledge when needed. "
    "If memory context is provided, prefer it and cite it."
)


def _pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_model_source(
    repo_id: str,
    cache_dir: Optional[str],
    force_download: bool,
) -> tuple[str, bool]:
    if Path(repo_id).exists():
        print(f"Using local model path: {repo_id}")
        return repo_id, True
    if snapshot_download is None:
        print("huggingface_hub not available; loading via transformers.")
        return repo_id, False
    if force_download:
        print("Force download enabled; downloading model files with per-file progress.")
        cache_path = snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True,
        )
        return cache_path, True
    try:
        cache_path = snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        print(f"Cache hit: model snapshot already present at {cache_path}")
        return cache_path, True
    except LocalEntryNotFoundError:
        print("Cache miss: downloading model files with per-file progress.")
        cache_path = snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True,
        )
        return cache_path, True
    except Exception:
        print("Cache lookup failed; downloading via transformers.")
        return repo_id, False


EXTRA_EVENT_PATHS: List[Path] = []
_RETRIEVAL_INDEX: Optional[storage.RetrievalIndex] = None


def _ensure_index(refresh: bool = False) -> storage.RetrievalIndex:
    global _RETRIEVAL_INDEX
    if _RETRIEVAL_INDEX is None or refresh:
        _RETRIEVAL_INDEX = storage.RetrievalIndex.build(
            extra_paths=EXTRA_EVENT_PATHS or None
        )
        storage.RETRIEVAL_INDEX = _RETRIEVAL_INDEX
    return _RETRIEVAL_INDEX


def _iter_event_texts(events_path: Path, passage_prefix: str) -> Sequence[tuple[str, str]]:
    for evt in storage.iter_normalized_events(events_path):
        event_id = evt.get("event_id")
        if not event_id:
            continue
        question = evt.get("question") or ""
        response = evt.get("response") or ""
        text = f"{passage_prefix}{question} {response}".strip()
        yield event_id, text


def _prepare_dataset() -> Optional[Path]:
    if not args.ace_dataset:
        raise SystemExit("ACE dataset ID not set. Provide --ace-dataset or ACE_HF_DATASET.")

    from datasets import load_dataset
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    events_path = storage.STORAGE.normalized_events / "ace_events_h1_2025.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    if not events_path.exists():
        ds = load_dataset(args.ace_dataset, args.ace_dataset_config, split="train")
        with events_path.open("w", encoding="utf-8") as handle:
            for row in tqdm(ds, desc="Writing dataset events", total=len(ds)):
                handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")
        print(f"Wrote dataset events: {events_path}")
    else:
        print(f"Dataset events already present: {events_path}")

    episode_dir = storage.STORAGE.episodes
    if not episode_dir.exists() or not any(episode_dir.rglob("*.json")):
        summary = storage.roll_up_episodes(
            events_path=events_path,
            destination_dir=episode_dir,
            overwrite=True,
            show_progress=True,
        )
        print("Episode roll-up:", summary)
    else:
        print(f"Episodes already present: {episode_dir}")

    embed_dir = storage.STORAGE.base_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    model_slug = DEFAULT_EMBED_MODEL_NAME.replace("/", "_").replace("-", "_")
    embed_path = embed_dir / f"event_embeddings_{model_slug}.npz"
    query_prefix = "query: "
    passage_prefix = "passage: "

    if args.rebuild_dataset_embeddings or not embed_path.exists():
        embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL_NAME)
        event_ids: List[str] = []
        vectors: List[np.ndarray] = []
        batch_ids: List[str] = []
        batch_texts: List[str] = []
        batch_size = 64
        total_events = sum(1 for _ in storage.iter_normalized_events(events_path))
        for event_id, text in tqdm(
            _iter_event_texts(events_path, passage_prefix),
            desc="Embedding events",
            total=total_events,
        ):
            batch_ids.append(event_id)
            batch_texts.append(text)
            if len(batch_texts) >= batch_size:
                emb = embed_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                vectors.append(emb)
                event_ids.extend(batch_ids)
                batch_ids.clear()
                batch_texts.clear()
        if batch_texts:
            emb = embed_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors.append(emb)
            event_ids.extend(batch_ids)
        if vectors:
            stacked = np.vstack(vectors)
            np.savez(embed_path, event_ids=np.array(event_ids), vectors=stacked)
            print(f"Wrote embeddings: {embed_path}")
    else:
        embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL_NAME)
        print(f"Using cached embeddings: {embed_path}")

    storage.load_event_embeddings_npz(embed_path)
    storage.register_event_embedder(
        lambda text: embed_model.encode([f"{query_prefix}{text}"], convert_to_numpy=True)[0]
    )
    print("Event embeddings loaded:", len(storage.EVENT_EMBEDDINGS))
    return events_path


def main() -> None:
    if args.use_ace_dataset:
        dataset_path = _prepare_dataset()
        if dataset_path is not None:
            EXTRA_EVENT_PATHS.append(dataset_path)
            _ensure_index(refresh=True)

    device = _pick_device(args.device)
    dtype = None
    if args.model_dtype == "float16":
        dtype = torch.float16
    elif args.model_dtype == "bfloat16":
        dtype = torch.bfloat16
    print(f"Loading model {args.model} on {device}...")
    if args.hf_cache_dir:
        print(f"Using Hugging Face cache dir: {args.hf_cache_dir}")
    model_source, local_only = _resolve_model_source(
        args.model,
        args.hf_cache_dir,
        args.force_download,
    )
    load_kwargs = {
        "cache_dir": args.hf_cache_dir,
        "force_download": args.force_download,
        "local_files_only": local_only,
    }
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    if args.low_cpu_mem:
        load_kwargs["low_cpu_mem_usage"] = True
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        **load_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **load_kwargs,
    )
    if device != "cpu":
        print(f"Moving model to {device}...")
        model.to(device)
    model.eval()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.top_k = 0
    print("Model ready")

    run_chat_loop(
        model=model,
        tokenizer=tokenizer,
        storage=storage,
        chat_id=args.chat_id,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        extract_max_new_tokens=args.extract_max_new_tokens,
        event_context_budget_tokens=args.event_context_budget,
        temperature=args.temperature,
        top_p=args.top_p,
        refresh_every=args.refresh_every,
        episode_lookback_months=args.episode_lookback_months,
        episode_events_per_context=args.episode_events_per_context,
        episode_top_k=args.episode_top_k,
        episode_prefilter=args.episode_prefilter,
        memory_score_threshold=args.memory_score_threshold,
        allow_general_fallback=args.allow_general_fallback,
        memory_system_prompt=MEMORY_SYSTEM_PROMPT,
        general_system_prompt=GENERAL_SYSTEM_PROMPT,
        stream=args.stream,
        echo_input=False,
    )


if __name__ == "__main__":
    main()
