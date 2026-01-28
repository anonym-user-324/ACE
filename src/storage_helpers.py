"""
Reusable helpers for the ACE time-aware storage system.

This module centralises the schema, ingestion, and episodic roll-up utilities
so other notebooks (e.g., a live RAG agent) can record and organise events
without re-running the original storage design notebook.
"""

from __future__ import annotations

import json
import os
import itertools
import math
import shutil
from collections import Counter
from collections.abc import Iterable as IterableABC, Mapping
import calendar
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Set

import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

# ---------------------------------------------------------------------------
# Paths & storage configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = ROOT_DIR / "data"
DATA_DIR = Path(os.getenv("ACE_STORAGE_DIR", DEFAULT_DATA_DIR)).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_BATCH_DIR = DATA_DIR / "raw_batches"

@dataclass(frozen=True)
class StorageConfig:
    base_dir: Path
    raw_batches: Path
    normalized_events: Path
    episodes: Path

    @classmethod
    def build(cls, base_dir: Path, raw_batches: Path) -> "StorageConfig":
        events_dir = base_dir / "events"
        episodes_dir = base_dir / "episodes"
        for path in (events_dir, episodes_dir):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            base_dir=base_dir,
            raw_batches=raw_batches,
            normalized_events=events_dir,
            episodes=episodes_dir,
        )


STORAGE = StorageConfig.build(base_dir=DATA_DIR, raw_batches=RAW_BATCH_DIR)


# ---------------------------------------------------------------------------
# Episode configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeConfig:
    months_to_keep: int = int(os.getenv("ACE_EPISODE_MONTHS", "12"))
    sub_bucket_days: int = int(os.getenv("ACE_EPISODE_SUB_BUCKET_DAYS", "7"))
    max_events_per_episode: int = int(os.getenv("ACE_EPISODE_MAX_EVENTS", "100"))
    manifest_name: str = os.getenv("ACE_EPISODE_MANIFEST", "index.json")

    @property
    def window_duration(self) -> timedelta:
        return timedelta(days=self.sub_bucket_days)


EPISODE_CONFIG = EpisodeConfig()


# ---------------------------------------------------------------------------
# Embedding caches (documented for RAG integration)
# ---------------------------------------------------------------------------

EVENT_EMBEDDINGS: Dict[str, np.ndarray] = {}
EVENT_EMBEDDER: Optional[Callable[[str], np.ndarray]] = None


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

EVENT_SCHEMA: Dict[str, Dict[str, Tuple[type, ...]]] = {
    "required": {
        "event_id": (str,),
        "thread_id": (str,),
        "timestamp": (str,),
        "ts_unix": (int, float),
        "question": (str,),
        "response": (str,),
        "keywords": (list,),
        "source_batch": (str,),
    },
    "optional": {
        "action_taken": (str,),
        "evidence_taken": (str,),
        "action_suggested": (str, list),
        "evidence_suggested": (str,),
        "source_suggested": (str,),
        "metadata": (dict,),
        "batch_request_id": (str,),
        "raw_custom_id": (str,),
    },
}


class EventValidationError(ValueError):
    """Raised when an event record fails validation."""


def _flatten_keywords(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        iterable = value.values()
    elif isinstance(value, IterableABC) and not isinstance(value, (str, bytes)):
        iterable = value
    else:
        return [value]
    flattened: List[Any] = []
    for item in iterable:
        flattened.extend(_flatten_keywords(item))
    return flattened


def _normalize_keywords(value: Any) -> Tuple[List[str], int]:
    flattened = _flatten_keywords(value)
    normalized: List[str] = []
    dropped = 0
    for item in flattened:
        text = str(item).strip()
        if text:
            normalized.append(text)
        else:
            dropped += 1
    return normalized, dropped


def _coerce_optional(value: Optional[Any]) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None
    return value


def _normalize_timestamp(ts_value: Any) -> Tuple[float, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}
    try:
        ts_float = float(ts_value)
    except Exception as exc:  # pragma: no cover - defensive
        raise EventValidationError(f"Invalid timestamp value: {ts_value}") from exc
    adjustments = 0
    while ts_float > 1e11:  # downscale ms/us timestamps
        ts_float /= 1000.0
        adjustments += 1
    if adjustments:
        diagnostics["timestamp_adjustments"] = adjustments
    return ts_float, diagnostics


def _isoformat_from_ts(ts_value: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts_value), tz=timezone.utc).isoformat()
    except Exception as exc:  # pragma: no cover - defensive
        raise EventValidationError(f"Invalid timestamp value: {ts_value}") from exc



def validate_event(event: Dict[str, Any]) -> None:
    """Validate the normalized event against the schema."""
    for field, types in EVENT_SCHEMA["required"].items():
        if field not in event:
            raise EventValidationError(f"missing required field: {field}")
        if not isinstance(event[field], types):
            raise EventValidationError(
                f"field {field} expected {types}, got {type(event[field])}"
            )
    if not isinstance(event["keywords"], list):
        raise EventValidationError("keywords field must be a list of strings")
    for idx, item in enumerate(event["keywords"]):
        if not isinstance(item, str):
            raise EventValidationError(f"keywords[{idx}] must be str, got {type(item)}")

    for field, types in EVENT_SCHEMA["optional"].items():
        value = event.get(field)
        if value is None:
            continue
        if not isinstance(value, types):
            raise EventValidationError(f"field {field} expected {types}, got {type(value)}")
        if isinstance(value, list):
            for idx, item in enumerate(value):
                if not isinstance(item, str):
                    raise EventValidationError(f"{field}[{idx}] must be str, got {type(item)}")


def normalize_event_payload(
    payload: Dict[str, Any],
    *,
    event_id: str,
    source_batch: str,
    request_id: Optional[str],
    custom_id: Optional[str],
) -> Dict[str, Any]:
    ts_raw = payload.get("ts")
    if ts_raw is None:
        raise EventValidationError(f"payload missing ts for event {event_id}")
    ts_seconds, ts_diag = _normalize_timestamp(ts_raw)
    metadata = payload.get("M") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw": metadata}
    keywords, keywords_dropped = _normalize_keywords(payload.get("K"))
    if keywords_dropped:
        diagnostics = metadata.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            metadata["diagnostics"] = diagnostics
        diagnostics["keywords_dropped"] = diagnostics.get("keywords_dropped", 0) + keywords_dropped
    if ts_diag:
        diagnostics = metadata.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            metadata["diagnostics"] = diagnostics
        for key, value in ts_diag.items():
            diagnostics[key] = diagnostics.get(key, 0) + value
    normalized: Dict[str, Any] = {
        "event_id": event_id,
        "thread_id": str(custom_id).split("-")[-1] if custom_id else event_id,
        "timestamp": _isoformat_from_ts(ts_seconds),
        "ts_unix": ts_seconds,
        "question": str(payload.get("q") or "").strip(),
        "response": str(payload.get("r") or "").strip(),
        "keywords": keywords,
        "action_taken": _coerce_optional(payload.get("action_taken")),
        "evidence_taken": _coerce_optional(payload.get("evidence_taken")),
        "action_suggested": _coerce_optional(payload.get("action_suggested")),
        "evidence_suggested": _coerce_optional(payload.get("evidence_suggested")),
        "source_suggested": _coerce_optional(payload.get("source_suggested")),
        "metadata": metadata,
        "source_batch": source_batch,
        "batch_request_id": request_id,
        "raw_custom_id": custom_id,
    }
    return normalized


def append_event(event: Dict[str, Any], *, storage: StorageConfig = STORAGE) -> Dict[str, Any]:
    """
    Append a single, fully-normalized event to the canonical store.

    The event must already satisfy ``EVENT_SCHEMA``.
    """
    validate_event(event)
    destination = storage.normalized_events / "custom_events.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    episode_info = update_episode_store(event)
    return {
        "destination": str(destination),
        "event_id": event["event_id"],
        "episode": episode_info,
    }


# ---------------------------------------------------------------------------
# Episodic consolidation
# ---------------------------------------------------------------------------

EPISODE_MAX_EVENTS_DEFAULT = EPISODE_CONFIG.max_events_per_episode
VALID_TS_THRESHOLD = 1e9  # ~2001-09-09 for distinguishing placeholder timestamps
UNKNOWN_BUCKET_ID = "unknown"
EPISODE_FILE_SUFFIX = ".json"


def iter_normalized_events(source_path: Path = STORAGE.normalized_events / "ace_events_h1_2025.jsonl") -> Iterator[Dict[str, Any]]:
    if not source_path.exists():
        return iter([])
    with source_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_events_in_file(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _month_key_from_ts(ts_value: float) -> Optional[Tuple[str, datetime]]:
    if ts_value is None or ts_value < VALID_TS_THRESHOLD:
        return None
    dt = datetime.fromtimestamp(float(ts_value), tz=timezone.utc)
    month_key = f"{dt.year}-{dt.month:02d}"
    return month_key, dt


def _slot_info(dt: datetime, config: EpisodeConfig) -> Tuple[int, str, datetime, datetime]:
    days_in_month = calendar.monthrange(dt.year, dt.month)[1]
    slot_index = min((dt.day - 1) // config.sub_bucket_days, max((days_in_month - 1) // config.sub_bucket_days, 0))
    slot_start_day = slot_index * config.sub_bucket_days + 1
    slot_end_day = min(slot_start_day + config.sub_bucket_days - 1, days_in_month)
    slot_start = datetime(dt.year, dt.month, slot_start_day, tzinfo=timezone.utc)
    slot_end = datetime(dt.year, dt.month, slot_end_day, 23, 59, 59, tzinfo=timezone.utc)
    slot_label = f"{dt.year}-{dt.month:02d}-w{slot_index + 1:02d}"
    return slot_index, slot_label, slot_start, slot_end


def _ensure_month_dir(base_dir: Path, month_key: str) -> Path:
    month_dir = base_dir / month_key
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir


def _episode_path(base_dir: Path, month_key: str, slot_label: str) -> Path:
    safe_label = slot_label.replace("/", "-")
    return base_dir / month_key / f"{safe_label}{EPISODE_FILE_SUFFIX}"


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _update_episode_summary(episode: Dict[str, Any]) -> None:
    events = episode.get("events", [])
    if not events:
        return
    events.sort(key=lambda evt: (evt.get("ts_unix") or 0.0, evt.get("event_id")))
    keywords = Counter()
    action_taken = 0
    action_suggested = 0
    chat_ids: Set[str] = set()
    for evt in events:
        keywords.update(evt.get("keywords", []))
        if evt.get("action_taken"):
            action_taken += 1
        if evt.get("action_suggested"):
            action_suggested += 1
        chat_id = evt.get("chat_id")
        if chat_id:
            chat_ids.add(chat_id)
    start_ts = events[0].get("ts_unix") or 0.0
    end_ts = events[-1].get("ts_unix") or start_ts
    episode["start_ts"] = start_ts
    episode["end_ts"] = end_ts
    episode["start_iso"] = _isoformat_from_ts(start_ts)
    episode["end_iso"] = _isoformat_from_ts(end_ts)
    dominant_keywords = [kw for kw, _ in keywords.most_common(8)]
    episode["dominant_keywords"] = dominant_keywords
    episode["action_counts"] = {
        "action_taken": action_taken,
        "action_suggested": action_suggested,
    }
    episode["summary"] = {
        "first_question": events[0].get("question"),
        "last_response": events[-1].get("response"),
        "keywords": dominant_keywords[:3],
    }
    metadata = episode.setdefault("metadata", {})
    metadata["event_count"] = len(events)
    metadata["chat_ids"] = sorted(chat_ids)
    episode["updated_at"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def update_episode_store(
    event: Dict[str, Any],
    *,
    episodes_base: Optional[Path] = None,
    config: EpisodeConfig = EPISODE_CONFIG,
) -> Dict[str, Any]:
    episodes_base = (episodes_base or STORAGE.episodes).resolve()
    ts_value = event.get("ts_unix")
    month_info = _month_key_from_ts(ts_value) if ts_value is not None else None
    if month_info is None:
        month_key = UNKNOWN_BUCKET_ID
        slot_label = f"{UNKNOWN_BUCKET_ID}-catchall"
        slot_start = datetime.utcnow().replace(tzinfo=timezone.utc)
        slot_end = slot_start + config.window_duration
    else:
        month_key, dt = month_info
        _, slot_label, slot_start, slot_end = _slot_info(dt, config)
    month_dir = _ensure_month_dir(episodes_base, month_key)
    episode_path = _episode_path(episodes_base, month_key, slot_label)
    episode = _load_json_file(episode_path)
    if not episode:
        episode = {
            "episode_id": slot_label,
            "month": month_key,
            "slot_label": slot_label,
            "slot_start": slot_start.isoformat(),
            "slot_end": slot_end.isoformat(),
            "events": [],
            "metadata": {"month": month_key, "slot_label": slot_label},
        }
    episode_events = episode.setdefault("events", [])
    episode_events.append(dict(event))
    _update_episode_summary(episode)
    _write_json_file(episode_path, episode)
    refresh_month_manifest(month_key, episodes_base=episodes_base, config=config)
    return {
        "episode_id": episode["episode_id"],
        "episode_path": str(episode_path),
        "month": month_key,
        "slot_label": slot_label,
        "event_count": len(episode_events),
    }


def refresh_month_manifest(
    month_key: str,
    *,
    episodes_base: Optional[Path] = None,
    config: EpisodeConfig = EPISODE_CONFIG,
) -> Dict[str, Any]:
    episodes_base = (episodes_base or STORAGE.episodes).resolve()
    month_dir = episodes_base / month_key
    if not month_dir.exists():
        return {}
    manifest_entries: List[Dict[str, Any]] = []
    for episode_file in sorted(month_dir.glob(f"*{EPISODE_FILE_SUFFIX}")):
        if episode_file.name == config.manifest_name:
            continue
        episode = _load_json_file(episode_file)
        if not episode:
            continue
        entry = {
            "episode_id": episode.get("episode_id"),
            "file": episode_file.name,
            "start_ts": episode.get("start_ts"),
            "end_ts": episode.get("end_ts"),
            "start_iso": episode.get("start_iso"),
            "end_iso": episode.get("end_iso"),
            "keywords": episode.get("dominant_keywords", []),
            "event_count": episode.get("metadata", {}).get("event_count", 0),
            "chat_ids": episode.get("metadata", {}).get("chat_ids", []),
            "first_question": episode.get("summary", {}).get("first_question"),
            "last_response": episode.get("summary", {}).get("last_response"),
        }
        manifest_entries.append(entry)
    manifest_entries.sort(key=lambda item: (item.get("start_ts") or 0.0, item["episode_id"]))
    manifest = {
        "month": month_key,
        "episodes": manifest_entries,
        "updated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    }
    manifest_path = month_dir / config.manifest_name
    _write_json_file(manifest_path, manifest)
    return manifest


def refresh_all_manifests(
    *,
    episodes_base: Optional[Path] = None,
    config: EpisodeConfig = EPISODE_CONFIG,
) -> Dict[str, Dict[str, Any]]:
    episodes_base = (episodes_base or STORAGE.episodes).resolve()
    manifests: Dict[str, Dict[str, Any]] = {}
    for month_dir in sorted(p for p in episodes_base.iterdir() if p.is_dir()):
        manifests[month_dir.name] = refresh_month_manifest(
            month_dir.name, episodes_base=episodes_base, config=config
        )
    return manifests


def query_events_by_time(
    start: datetime,
    end: datetime,
    *,
    chat_id: Optional[str] = None,
    episodes_base: Optional[Path] = None,
    limit_episodes: Optional[int] = None,
    events_per_episode: Optional[int] = None,
    config: EpisodeConfig = EPISODE_CONFIG,
) -> List[Dict[str, Any]]:
    episodes_base = (episodes_base or STORAGE.episodes).resolve()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)
    if end < start:
        start, end = end, start
    start_ts = start.timestamp()
    end_ts = end.timestamp()

    results: List[Dict[str, Any]] = []
    for month_dir in sorted((p for p in episodes_base.iterdir() if p.is_dir())):
        month_key = month_dir.name
        manifest_path = month_dir / config.manifest_name
        if not manifest_path.exists():
            manifest = refresh_month_manifest(month_key, episodes_base=episodes_base, config=config)
        else:
            manifest = _load_json_file(manifest_path)
        for entry in manifest.get("episodes", []):
            ep_start = entry.get("start_ts") or 0.0
            ep_end = entry.get("end_ts") or ep_start
            if ep_end < start_ts or ep_start > end_ts:
                continue
            if chat_id and chat_id not in entry.get("chat_ids", []):
                continue
            episode_path = month_dir / entry["file"]
            episode = _load_json_file(episode_path)
            if not episode:
                continue
            events = episode.get("events", [])
            matching_events = [
                evt
                for evt in events
                if start_ts <= (evt.get("ts_unix") or 0.0) <= end_ts
            ]
            if not matching_events:
                continue
            if events_per_episode and len(matching_events) > events_per_episode:
                matching_events = matching_events[-events_per_episode:]
            results.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "month": month_key,
                    "episode_path": str(episode_path),
                    "start_ts": entry.get("start_ts"),
                    "end_ts": entry.get("end_ts"),
                    "events": matching_events,
                    "chat_ids": entry.get("chat_ids", []),
                    "keywords": entry.get("keywords", []),
                    "summary": episode.get("summary", {}),
                    "score": float(len(matching_events)),
                }
            )
    results.sort(key=lambda item: item["events"][-1].get("ts_unix") or 0.0, reverse=True)
    if limit_episodes is not None:
        results = results[:limit_episodes]
    return results


def retrieve_episode_context(
    query: str,
    *,
    limit: int = 3,
    months: Optional[int] = None,
    chat_id: Optional[str] = None,
    events_per_episode: int = 5,
    episodes_base: Optional[Path] = None,
    config: EpisodeConfig = EPISODE_CONFIG,
) -> List[Dict[str, Any]]:
    episodes_base = (episodes_base or STORAGE.episodes).resolve()
    tokens = {token for token in query.lower().split() if token}
    month_dirs = sorted(
        (p for p in episodes_base.iterdir() if p.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    if months is not None:
        month_dirs = month_dirs[:months]
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for month_dir in month_dirs:
        month_key = month_dir.name
        manifest_path = month_dir / config.manifest_name
        if not manifest_path.exists():
            manifest = refresh_month_manifest(month_key, episodes_base=episodes_base, config=config)
        else:
            manifest = _load_json_file(manifest_path)
        for entry in manifest.get("episodes", []):
            if chat_id and chat_id not in entry.get("chat_ids", []):
                continue
            keywords = {kw.lower() for kw in entry.get("keywords", []) if kw}
            lexical = len(tokens & keywords) if tokens else 0.0
            if lexical == 0.0 and tokens:
                snippet_tokens = set(
                    ((entry.get("first_question") or "") + " " + (entry.get("last_response") or "")).lower().split()
                )
                lexical = len(tokens & snippet_tokens)
            recency = (entry.get("end_ts") or 0.0) * 1e-9
            score = lexical + 0.1 * len(entry.get("chat_ids", [])) + recency
            scored.append((score, month_key, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    results: List[Dict[str, Any]] = []
    for score, month_key, entry in itertools.islice(scored, limit):
        episode_path = episodes_base / month_key / entry["file"]
        episode = _load_json_file(episode_path)
        if not episode:
            continue
        events_payload = episode.get("events", [])
        if events_per_episode is not None and events_per_episode > 0:
            events_slice = events_payload[-events_per_episode:]
        else:
            events_slice = events_payload
        results.append(
            {
                "score": score,
                "month": month_key,
                "episode_id": entry.get("episode_id"),
                "episode_path": str(episode_path),
                "start_ts": entry.get("start_ts"),
                "end_ts": entry.get("end_ts"),
                "keywords": entry.get("keywords", []),
                "chat_ids": entry.get("chat_ids", []),
                "events": events_slice,
                "summary": episode.get("summary", {}),
            }
        )
    return results


def prune_old_episodes(
    *,
    episodes_base: Optional[Path] = None,
    config: EpisodeConfig = EPISODE_CONFIG,
) -> Dict[str, Any]:
    episodes_base = (episodes_base or STORAGE.episodes).resolve()
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    cutoff_month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    removed: List[str] = []
    for month_dir in episodes_base.iterdir():
        if not month_dir.is_dir():
            continue
        if month_dir.name == UNKNOWN_BUCKET_ID:
            continue
        try:
            year, month = (int(part) for part in month_dir.name.split("-"))
        except (ValueError, TypeError):
            continue
        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
        months_delta = (cutoff_month_start.year - month_start.year) * 12 + (
            cutoff_month_start.month - month_start.month
        )
        if months_delta >= config.months_to_keep:
            for child in month_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            month_dir.rmdir()
            removed.append(str(month_dir))
    return {"removed": removed, "months_to_keep": config.months_to_keep}


def roll_up_episodes(
    events_path: Path = STORAGE.normalized_events / "ace_events_h1_2025.jsonl",
    *,
    destination_dir: Path = STORAGE.episodes,
    config: EpisodeConfig = EPISODE_CONFIG,
    overwrite: bool = True,
    skip_existing: bool = False,
    show_progress: bool = False,
) -> Dict[str, Any]:
    events = list(iter_normalized_events(events_path))
    if not events:
        return {}
    if overwrite and destination_dir.exists():
        for child in destination_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    destination_dir.mkdir(parents=True, exist_ok=True)
    existing_event_ids: Set[str] = set()
    if skip_existing and not overwrite:
        for episode_file in destination_dir.rglob(f"*{EPISODE_FILE_SUFFIX}"):
            if episode_file.name == config.manifest_name:
                continue
            episode = _load_json_file(episode_file)
            if not episode:
                continue
            for evt in episode.get("events", []):
                event_id = evt.get("event_id")
                if event_id:
                    existing_event_ids.add(str(event_id))
    month_counts: Counter[str] = Counter()
    iterable = events
    if show_progress and tqdm is not None:
        iterable = tqdm(events, desc="Building episodes", total=len(events))
    for event in iterable:
        event_id = event.get("event_id")
        if event_id and event_id in existing_event_ids:
            continue
        info = update_episode_store(event, episodes_base=destination_dir, config=config)
        month_counts[info["month"]] += 1
        if event_id:
            existing_event_ids.add(str(event_id))
    manifests = refresh_all_manifests(episodes_base=destination_dir, config=config)
    summary = {
        "total_events": len(events),
        "months": dict(month_counts),
        "manifest_count": len(manifests),
        "destination": str(destination_dir),
    }
    return summary


def build_episodes_from_jsonl(
    events_path: Path,
    *,
    destination_dir: Path = STORAGE.episodes,
    config: EpisodeConfig = EPISODE_CONFIG,
    overwrite: bool = True,
    skip_existing: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build episode files from a normalized events JSONL file."""
    if skip_existing is None:
        skip_existing = not overwrite
    return roll_up_episodes(
        events_path=events_path,
        destination_dir=destination_dir,
        config=config,
        overwrite=overwrite,
        skip_existing=skip_existing,
    )


# ---------------------------------------------------------------------------
# Retrieval scaffolding (dense + lexical + time-aware scoring)
# ---------------------------------------------------------------------------

try:  # optional dependency; fall back to numpy if unavailable
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional
    faiss = None


def register_event_embeddings(
    embeddings: Dict[str, np.ndarray],
    *,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Register precomputed event embeddings for retrieval scoring."""
    EVENT_EMBEDDINGS.clear()
    normalized: Dict[str, np.ndarray] = {}
    for event_id, vector in embeddings.items():
        arr = np.asarray(vector, dtype=np.float32)
        if normalize:
            norm = float(np.linalg.norm(arr))
            if norm:
                arr = arr / norm
        EVENT_EMBEDDINGS[event_id] = arr
        normalized[event_id] = arr
    return normalized


def load_event_embeddings_npz(
    path: Path,
    *,
    register: bool = True,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Load event embeddings from an npz file (with vectors + event_ids arrays)."""
    data = np.load(str(path), allow_pickle=False)
    vectors = data["vectors"]
    event_ids = data["event_ids"]
    mapping: Dict[str, np.ndarray] = {}
    for idx, event_id in enumerate(event_ids):
        mapping[str(event_id)] = vectors[idx]
    if register:
        register_event_embeddings(mapping, normalize=normalize)
    return mapping


def register_event_embedder(func: Callable[[str], np.ndarray]) -> None:
    """Register a callable that embeds query text for retrieval scoring."""
    global EVENT_EMBEDDER
    EVENT_EMBEDDER = func


def _time_decay(now_ts: float, ts: float, half_life_days: float) -> float:
    """Exponential decay factor based on age in days."""
    if half_life_days <= 0:
        return 1.0
    delta_days = max(0.0, (now_ts - ts) / 86400.0)
    return 0.5 ** (delta_days / half_life_days)


@dataclass
class RetrievalIndex:
    events: List[Dict[str, Any]]
    timestamps: List[float]
    action_mask: List[bool]
    keyword_cache: List[set]
    dense_vectors: Optional[np.ndarray]
    faiss_index: Optional[Any]
    dense_ids: List[int]
    half_life_days: float
    lexical_weight: float
    time_weight: float
    dense_candidates: int

    @classmethod
    def build(
        cls,
        primary_path: Path = STORAGE.normalized_events / "custom_events.jsonl",
        extra_paths: Optional[List[Path]] = None,
        *,
        half_life_days: float = float(os.getenv("ACE_TIME_DECAY_HALF_LIFE_DAYS", "30")),
        lexical_weight: float = float(os.getenv("ACE_RETRIEVE_LEXICAL_WEIGHT", "0.1")),
        time_weight: float = float(os.getenv("ACE_RETRIEVE_TIME_WEIGHT", "0.2")),
        dense_candidates: int = int(os.getenv("ACE_RETRIEVE_DENSE_CANDIDATES", "1000")),
        show_progress: bool = False,
    ) -> "RetrievalIndex":
        events: List[Dict[str, Any]] = []
        for path in [primary_path] + (extra_paths or []):
            if path.exists():
                events.extend(iter_normalized_events(path))
        events.sort(key=lambda evt: evt["ts_unix"])
        action_mask: List[bool] = []
        keyword_cache: List[set] = []
        vectors: List[np.ndarray] = []
        dense_ids: List[int] = []
        embedding_lookup = EVENT_EMBEDDINGS or {}
        iterable = events
        if show_progress and tqdm is not None:
            iterable = tqdm(events, desc="Building retrieval index", total=len(events))
        for idx, event in enumerate(iterable):
            action_mask.append(bool(event.get("action_taken") or event.get("action_suggested")))
            keyword_cache.append({token.lower() for token in event.get("keywords", [])})
            event_id = event.get("event_id")
            vec = embedding_lookup.get(event_id) if event_id else None
            if vec is not None:
                vectors.append(np.asarray(vec, dtype=np.float32))
                dense_ids.append(idx)
        timestamps = [evt["ts_unix"] for evt in events]
        dense_vectors: Optional[np.ndarray] = None
        dense_index = None
        if vectors:
            dense_vectors = np.stack(vectors, axis=0)
            norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            dense_vectors = dense_vectors / norms
            if faiss is not None:  # pragma: no cover - optional dependency
                dim = dense_vectors.shape[1]
                dense_index = faiss.IndexHNSWFlat(dim, 32)
                dense_index.hnsw.efConstruction = 80
                dense_index.hnsw.efSearch = max(50, dense_candidates)
                dense_index.add(dense_vectors)
        return cls(
            events=events,
            timestamps=timestamps,
            action_mask=action_mask,
            keyword_cache=keyword_cache,
            dense_vectors=dense_vectors,
            faiss_index=dense_index,
            dense_ids=dense_ids,
            half_life_days=half_life_days,
            lexical_weight=lexical_weight,
            time_weight=time_weight,
            dense_candidates=dense_candidates,
        )

    def _dense_scores(self, query_vector: np.ndarray, top_k: int) -> Dict[int, float]:
        """Return dense similarity scores mapped to event indices."""
        if self.faiss_index is not None:  # pragma: no cover - optional
            vec = query_vector.astype(np.float32)
            vec = vec / (np.linalg.norm(vec) or 1.0)
            scores, idxs = self.faiss_index.search(vec[None, :], top_k)
            mapping: Dict[int, float] = {}
            for score, local_idx in zip(scores[0].tolist(), idxs[0].tolist()):
                if local_idx < 0 or local_idx >= len(self.dense_ids):
                    continue
                event_idx = self.dense_ids[local_idx]
                mapping[event_idx] = float(score)
            return mapping
        if self.dense_vectors is None:
            return {}
        vec = query_vector.astype(np.float32)
        vec = vec / (np.linalg.norm(vec) or 1.0)
        sims = self.dense_vectors @ vec
        top_idx = np.argsort(sims)[::-1][:top_k]
        return {self.dense_ids[i]: float(sims[i]) for i in top_idx}

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        require_action: bool = False,
        time_window: Optional[Tuple[float, float]] = None,
        allowed_event_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval: dense (FAISS/np) + lexical + exponential time decay."""
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "for",
            "from",
            "has",
            "have",
            "i",
            "if",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "was",
            "we",
            "what",
            "when",
            "where",
            "why",
            "with",
            "you",
            "your",
        }
        query_tokens = {
            token
            for token in query.lower().split()
            if token and token not in stopwords and len(token) > 2
        }
        min_lexical_hit = int(os.getenv("ACE_RETRIEVE_MIN_LEXICAL_HIT", "1"))
        now_ts = datetime.utcnow().replace(tzinfo=timezone.utc).timestamp()
        allowed_idxs: Optional[Set[int]] = None
        if allowed_event_ids is not None:
            allowed_ids = {event_id for event_id in allowed_event_ids if event_id}
            if not allowed_ids:
                return []
            allowed_idxs = {
                idx
                for idx, evt in enumerate(self.events)
                if evt.get("event_id") in allowed_ids
            }
            if not allowed_idxs:
                return []

        query_vector: Optional[np.ndarray] = None
        if EVENT_EMBEDDER is not None:
            try:
                raw_vec = EVENT_EMBEDDER(query)
            except Exception:
                raw_vec = None
            if raw_vec is not None:
                vec = np.asarray(raw_vec, dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm:
                    vec = vec / norm
                query_vector = vec

        dense_scores = (
            self._dense_scores(query_vector, max(limit * 5, self.dense_candidates))
            if query_vector is not None
            else {}
        )
        candidate_idxs = set(dense_scores.keys()) if dense_scores else set(range(len(self.events)))
        if allowed_idxs is not None:
            candidate_idxs &= allowed_idxs
            if not candidate_idxs:
                candidate_idxs = set(allowed_idxs)
        if dense_scores and query_tokens:
            max_lexical = 0
            for idx in candidate_idxs:
                evt = self.events[idx]
                text_tokens = set(
                    (evt.get("question", "") + " " + evt.get("response", "")).lower().split()
                )
                text_tokens.update(self.keyword_cache[idx])
                max_lexical = max(max_lexical, len(query_tokens & text_tokens))
                if max_lexical >= min_lexical_hit:
                    break
            if max_lexical < min_lexical_hit:
                candidate_idxs = set(allowed_idxs) if allowed_idxs is not None else set(range(len(self.events)))

        start_ts, end_ts = (None, None)
        if time_window is not None:
            start_ts, end_ts = time_window

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for idx in candidate_idxs:
            evt = self.events[idx]
            if require_action and not self.action_mask[idx]:
                continue
            ts = self.timestamps[idx]
            if start_ts is not None and end_ts is not None and not (start_ts <= ts <= end_ts):
                continue

            text_tokens = set(
                (evt.get("question", "") + " " + evt.get("response", "")).lower().split()
            )
            text_tokens.update(self.keyword_cache[idx])
            lexical = len(query_tokens & text_tokens)

            dense = dense_scores.get(idx, 0.0)
            decay = _time_decay(now_ts, ts, self.half_life_days)
            effective_time_weight = 0.0 if time_window is not None else self.time_weight
            score = float(dense) + self.lexical_weight * float(lexical) + effective_time_weight * float(decay)
            evt_copy = dict(evt)
            evt_copy["score"] = score
            scored.append((score, evt_copy))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [event for _, event in itertools.islice(scored, limit)]


RETRIEVAL_INDEX: Optional[RetrievalIndex] = None


def ensure_retrieval_index(refresh: bool = False) -> RetrievalIndex:
    global RETRIEVAL_INDEX
    if RETRIEVAL_INDEX is None or refresh:
        events_dir = STORAGE.normalized_events
        extra_paths = []
        if events_dir.exists():
            for path in sorted(events_dir.glob("*.jsonl")):
                if path.name == "custom_events.jsonl":
                    continue
                extra_paths.append(path)
        RETRIEVAL_INDEX = RetrievalIndex.build(extra_paths=extra_paths or None)
    return RETRIEVAL_INDEX


def retrieve(
    query: str,
    *,
    limit: int = 5,
    require_action: bool = False,
    time_window: Optional[Tuple[float, float]] = None,
    allowed_event_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    index = ensure_retrieval_index()
    return index.search(
        query,
        limit=limit,
        require_action=require_action,
        time_window=time_window,
        allowed_event_ids=allowed_event_ids,
    )


__all__ = [
    "STORAGE",
    "StorageConfig",
    "EpisodeConfig",
    "EPISODE_CONFIG",
    "EVENT_SCHEMA",
    "EventValidationError",
    "normalize_event_payload",
    "validate_event",
    "iter_normalized_events",
    "append_event",
    "build_episodes_from_jsonl",
    "roll_up_episodes",
    "update_episode_store",
    "refresh_month_manifest",
    "refresh_all_manifests",
    "register_event_embeddings",
    "load_event_embeddings_npz",
    "register_event_embedder",
    "query_events_by_time",
    "retrieve_episode_context",
    "prune_old_episodes",
    "retrieve",
    "ensure_retrieval_index",
]
