from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text)


def summarize_with_budget(
    lines: List[str],
    *,
    budget_tokens: int,
    tokenizer=None,
) -> str:
    if budget_tokens <= 0:
        return "\n".join(lines) if lines else "- None"
    kept: List[str] = []
    total = 0
    for line in lines:
        candidate = line if not kept else "\n".join(kept + [line])
        candidate_tokens = _count_tokens(candidate, tokenizer)
        if candidate_tokens > budget_tokens and kept:
            break
        kept.append(line)
        total = candidate_tokens
        if total >= budget_tokens:
            break
    return "\n".join(kept) if kept else "- None"
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - optional
    ZoneInfo = None

from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

USER_TIMEZONE = os.getenv("ACE_USER_TZ")
if USER_TIMEZONE and ZoneInfo is not None:
    LOCAL_TZ = ZoneInfo(USER_TIMEZONE)
else:
    LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers",
    "him", "his", "i", "if", "in", "is", "it", "its", "me", "my", "no", "not",
    "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "they",
    "this", "to", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your", "yesterday", "today",
}

ACTION_SUGGEST_INSTRUCTIONS = (
    "You are annotating Time-Aware Episodic Memory.\n"
    "From the ASSISTANT response only, extract the single best action_suggested "
    "(imperative phrasing if possible) and a supporting evidence_suggested sentence "
    "verbatim from the response.\n"
    "Return strict JSON only with keys: action_suggested, evidence_suggested."
)

ACTION_TAKEN_SUMMARY_INSTRUCTIONS = (
    "Summarize the user's completed action as a short action phrase (no subject), "
    "5-10 words max. Return strict JSON only with key: action_taken."
)


def build_action_suggest_messages(response: str) -> List[Dict[str, str]]:
    content = (
        "Assistant response:\n"
        f"{response}\n\n"
        "Return strict JSON only."
    )
    return [
        {"role": "system", "content": ACTION_SUGGEST_INSTRUCTIONS},
        {"role": "user", "content": content},
    ]


def build_action_taken_messages(action_sentence: str) -> List[Dict[str, str]]:
    content = (
        "User action sentence:\n"
        f"{action_sentence}\n\n"
        "Return strict JSON only."
    )
    return [
        {"role": "system", "content": ACTION_TAKEN_SUMMARY_INSTRUCTIONS},
        {"role": "user", "content": content},
    ]


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        return match.group(0)
    return None


def parse_action_evidence(text: str) -> Dict[str, Optional[str]]:
    payload: Dict[str, Optional[str]] = {
        "action_taken": None,
        "evidence_taken": None,
        "action_suggested": None,
        "evidence_suggested": None,
    }
    if not text:
        return payload
    candidate = _extract_json_object(text)
    if candidate is None:
        return payload
    try:
        data = json.loads(candidate)
    except Exception:
        return payload

    def _clean(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            value = "; ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or cleaned.lower() in {"null", "none", "n/a"}:
                return None
            return cleaned
        return str(value).strip()

    for key in payload:
        payload[key] = _clean(data.get(key))
    return payload


def parse_action_suggest(text: str) -> Dict[str, Optional[str]]:
    payload: Dict[str, Optional[str]] = {
        "action_suggested": None,
        "evidence_suggested": None,
    }
    if not text:
        return payload
    candidate = _extract_json_object(text)
    if candidate is None:
        return payload
    try:
        data = json.loads(candidate)
    except Exception:
        return payload

    def _clean(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            value = "; ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or cleaned.lower() in {"null", "none", "n/a"}:
                return None
            return cleaned
        return str(value).strip()

    payload["action_suggested"] = _clean(data.get("action_suggested"))
    payload["evidence_suggested"] = _clean(data.get("evidence_suggested"))
    return payload


def parse_action_taken(text: str) -> Optional[str]:
    if not text:
        return None
    candidate = _extract_json_object(text)
    if candidate is None:
        cleaned = text.strip()
        return cleaned or None
    try:
        data = json.loads(candidate)
    except Exception:
        return None
    value = data.get("action_taken")
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"null", "none", "n/a"}:
            return None
        return cleaned
    return None


def select_action_sentence(query: str) -> Optional[str]:
    if not query:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", query.strip())
    if not sentences:
        sentences = [query.strip()]
    pattern = re.compile(
        r"\bI\s+(tried|attempted|used|ran|installed|updated|upgraded|downgraded|"
        r"changed|modified|added|removed|deleted|fixed|checked|tested|configured|"
        r"set|set up|setup|built|created|made|switched|enabled|disabled|"
        r"restarted|rebooted|reinstalled|followed)\b[^.?!]*",
        re.IGNORECASE,
    )
    for sentence in sentences:
        if pattern.search(sentence):
            return sentence.strip()
    return None

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

NUM_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclass
class TimeWindow:
    start: datetime
    end: datetime
    label: str
    confidence: float
    source: str = "auto"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "source": self.source,
            "start_iso": self.start.isoformat(),
            "end_iso": self.end.isoformat(),
        }


def _day_bounds(dt: datetime, tz: Optional[timezone] = None) -> Tuple[datetime, datetime]:
    tz = tz or LOCAL_TZ
    local = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)
    day_start = datetime(local.year, local.month, local.day, tzinfo=tz)
    day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
    return day_start, day_end


def _month_bounds(year: int, month: int, tz: Optional[timezone] = None) -> Tuple[datetime, datetime]:
    tz = tz or LOCAL_TZ
    start = datetime(year, month, 1, tzinfo=tz)
    end = start + relativedelta(months=1) - timedelta(seconds=1)
    return start, end


def _year_bounds(year: int, tz: Optional[timezone] = None) -> Tuple[datetime, datetime]:
    tz = tz or LOCAL_TZ
    start = datetime(year, 1, 1, tzinfo=tz)
    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=tz)
    return start, end


def _last_week_bounds(now: datetime, tz: Optional[timezone] = None) -> Tuple[datetime, datetime]:
    tz = tz or LOCAL_TZ
    local = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=tz)
    current_week_start = datetime(local.year, local.month, local.day, tzinfo=tz) - timedelta(days=local.weekday())
    last_week_end = current_week_start - timedelta(seconds=1)
    last_week_start = last_week_end - timedelta(days=6)
    return last_week_start.replace(hour=0, minute=0, second=0), last_week_end.replace(hour=23, minute=59, second=59)


def _last_month_bounds(now: datetime, tz: Optional[timezone] = None) -> Tuple[datetime, datetime]:
    tz = tz or LOCAL_TZ
    local = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=tz)
    first_of_month = datetime(local.year, local.month, 1, tzinfo=tz)
    last_month_end = first_of_month - timedelta(seconds=1)
    last_month_start = datetime(last_month_end.year, last_month_end.month, 1, tzinfo=tz)
    return last_month_start, last_month_end.replace(hour=23, minute=59, second=59)


def _parse_quantity(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    return NUM_WORDS.get(token)


def _parse_relative_window(lowered: str, now: datetime) -> Optional[TimeWindow]:
    rel_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(day|week|month|year)s?\s+ago\b", lowered)
    if not rel_match:
        return None
    qty = _parse_quantity(rel_match.group(1))
    unit = rel_match.group(2)
    if qty is None:
        return None
    if unit == "day":
        target = now - timedelta(days=qty)
        start, end = _day_bounds(target, tz=LOCAL_TZ)
        label = f"days_ago:{qty}"
    elif unit == "week":
        target = now - timedelta(weeks=qty)
        week_start = datetime(target.year, target.month, target.day, tzinfo=LOCAL_TZ) - timedelta(days=target.weekday())
        start = week_start.replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=7) - timedelta(seconds=1)
        label = f"weeks_ago:{qty}"
    elif unit == "month":
        target = now - relativedelta(months=qty)
        start, end = _month_bounds(target.year, target.month, tz=LOCAL_TZ)
        label = f"months_ago:{qty}"
    else:
        target = now - relativedelta(years=qty)
        start, end = _year_bounds(target.year, tz=LOCAL_TZ)
        label = f"years_ago:{qty}"
    return TimeWindow(start=start, end=end, label=label, confidence=0.85)


def detect_time_request(query: str, *, now: Optional[datetime] = None) -> Optional[TimeWindow]:
    if not query or not query.strip():
        return None
    now = now or datetime.now(tz=LOCAL_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=LOCAL_TZ)
    else:
        now = now.astimezone(LOCAL_TZ)
    text = query.strip()
    lowered = text.lower()

    range_match = re.search(r"(?:from|between)\s+(.+?)\s+(?:to|and)\s+(.+)", text, flags=re.IGNORECASE)
    if range_match:
        try:
            start_raw = date_parser.parse(range_match.group(1), default=now)
            end_raw = date_parser.parse(range_match.group(2), default=start_raw)
            if start_raw.tzinfo is None:
                start_raw = start_raw.replace(tzinfo=LOCAL_TZ)
            else:
                start_raw = start_raw.astimezone(LOCAL_TZ)
            if end_raw.tzinfo is None:
                end_raw = end_raw.replace(tzinfo=LOCAL_TZ)
            else:
                end_raw = end_raw.astimezone(LOCAL_TZ)
            if end_raw < start_raw:
                start_raw, end_raw = end_raw, start_raw
            return TimeWindow(start=start_raw, end=end_raw, label="range", confidence=0.9)
        except Exception:
            pass

    month_year_match = re.search(r"\b([A-Za-z]{3,9})\s+(\d{4})\b", lowered)
    if month_year_match:
        month_token = month_year_match.group(1)
        year = int(month_year_match.group(2))
        month = MONTHS.get(month_token)
        if month:
            start, end = _month_bounds(year, month, tz=LOCAL_TZ)
            return TimeWindow(start=start, end=end, label=f"month:{year}-{month:02d}", confidence=0.9)

    month_numeric = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", lowered)
    if month_numeric:
        year = int(month_numeric.group(1))
        month = int(month_numeric.group(2))
        start, end = _month_bounds(year, month, tz=LOCAL_TZ)
        return TimeWindow(start=start, end=end, label=f"month:{year}-{month:02d}", confidence=0.9)

    rel_window = _parse_relative_window(lowered, now)
    if rel_window:
        return rel_window

    if "last year" in lowered:
        start, end = _year_bounds(now.year - 1, tz=LOCAL_TZ)
        return TimeWindow(start=start, end=end, label="last_year", confidence=0.8)

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        try:
            parsed = date_parser.parse(iso_match.group(1))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=LOCAL_TZ)
            else:
                parsed = parsed.astimezone(LOCAL_TZ)
            start, end = _day_bounds(parsed, tz=LOCAL_TZ)
            return TimeWindow(start=start, end=end, label=f"date:{start.date()}", confidence=0.98)
        except Exception:
            pass

    explicit_match = re.search(
        r"(?:on|from|since)\s+([A-Za-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_match:
        try:
            parsed = date_parser.parse(explicit_match.group(1), default=now)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=LOCAL_TZ)
            else:
                parsed = parsed.astimezone(LOCAL_TZ)
            start, end = _day_bounds(parsed, tz=LOCAL_TZ)
            return TimeWindow(start=start, end=end, label=f"explicit:{start.date()}", confidence=0.95)
        except Exception:
            pass

    if "yesterday" in lowered or "last night" in lowered:
        start, end = _day_bounds(now - timedelta(days=1), tz=LOCAL_TZ)
        return TimeWindow(start=start, end=end, label="yesterday", confidence=0.9)

    if "today" in lowered or "earlier today" in lowered:
        start, end = _day_bounds(now, tz=LOCAL_TZ)
        return TimeWindow(start=start, end=end, label="today", confidence=0.6)

    if "last week" in lowered:
        start, end = _last_week_bounds(now, tz=LOCAL_TZ)
        return TimeWindow(start=start, end=end, label="last_week", confidence=0.85)

    if "last month" in lowered:
        start, end = _last_month_bounds(now, tz=LOCAL_TZ)
        return TimeWindow(start=start, end=end, label="last_month", confidence=0.8)

    return None


def extract_keywords(text: str, limit: int = 8) -> List[str]:
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _ in ranked[:limit]]


def summarize_events(
    event_hits: List[Dict[str, Any]],
    *,
    budget_tokens: Optional[int] = None,
    tokenizer=None,
) -> str:
    if budget_tokens is None:
        budget_tokens = int(os.getenv("ACE_EVENT_CONTEXT_BUDGET_TOKENS", "1200"))
    lines: List[str] = []
    for hit in event_hits:
        timestamp = _format_timestamp(hit.get("timestamp") or hit.get("ts_unix"))
        question = (hit.get("question") or "").strip()[:160]
        response = (hit.get("response") or "").strip()[:160]
        action_taken = hit.get("action_taken")
        if isinstance(action_taken, list):
            action_taken = "; ".join(str(item).strip() for item in action_taken if str(item).strip())
        action_taken = (action_taken or "").strip()

        action_suggested = hit.get("action_suggested")
        if isinstance(action_suggested, list):
            action_suggested = "; ".join(
                str(item).strip() for item in action_suggested if str(item).strip()
            )
        action_suggested = (action_suggested or "").strip()

        extras: List[str] = []
        if action_taken:
            extras.append(f"Action taken: {action_taken[:160]}")
        if action_suggested:
            extras.append(f"Action suggested: {action_suggested[:160]}")
        extra_text = f" | {' | '.join(extras)}" if extras else ""

        line = (
            f"- ({hit.get('score', 0.0):.3f}) [{timestamp}] "
            f"Q: {question} | A: {response}{extra_text}"
        )
        lines.append(line)
    return summarize_with_budget(lines, budget_tokens=budget_tokens, tokenizer=tokenizer)


def trim_episode_hits(
    episode_hits: List[Dict[str, Any]],
    *,
    events_per_episode: int,
) -> List[Dict[str, Any]]:
    trimmed: List[Dict[str, Any]] = []
    for hit in episode_hits:
        hit_copy = dict(hit)
        events = hit_copy.get("events", [])
        if events_per_episode is not None and events_per_episode > 0 and len(events) > events_per_episode:
            hit_copy["events"] = events[-events_per_episode:]
        trimmed.append(hit_copy)
    return trimmed


def _format_timestamp(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            text = str(value).strip()
            if not text:
                return "unknown"
            try:
                dt = date_parser.parse(text)
            except Exception:
                return text
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        iso = dt.astimezone(timezone.utc).isoformat()
        human = dt.astimezone(LOCAL_TZ).strftime("%b %d, %Y %H:%M %Z")
        return f"{iso} | {human}"
    except Exception:
        return str(value)


def summarize_episode_hits(
    episode_hits: List[Dict[str, Any]],
    *,
    window: Optional[TimeWindow] = None,
    summary_budget_tokens: Optional[int] = None,
    transcript_budget_tokens: Optional[int] = None,
    tokenizer=None,
) -> Tuple[str, str]:
    if summary_budget_tokens is None:
        summary_budget_tokens = int(os.getenv("ACE_EPISODE_SUMMARY_BUDGET_TOKENS", "800"))
    if transcript_budget_tokens is None:
        transcript_budget_tokens = int(os.getenv("ACE_EPISODE_TRANSCRIPT_BUDGET_TOKENS", "1200"))

    summary_lines: List[str] = []
    transcript_lines: List[str] = []
    if window is not None:
        summary_lines.append(
            f"*Time window* {window.label} ({window.start.isoformat()} -> {window.end.isoformat()})"
        )
    for hit in episode_hits:
        summary_lines.append(
            "- ({score:.3f}) {ep} [{month}] events={count}".format(
                score=hit.get("score", 0.0),
                ep=hit.get("episode_id", "unknown"),
                month=hit.get("month", "unknown"),
                count=len(hit.get("events", [])),
            )
        )
        for evt in hit.get("events", []):
            timestamp = _format_timestamp(evt.get("timestamp") or evt.get("ts_unix"))
            question = (evt.get("question") or "").strip()
            response = (evt.get("response") or "").strip()
            transcript_lines.append(f"[{timestamp}] Q: {question}\nA: {response}")

    summary_text = summarize_with_budget(
        summary_lines,
        budget_tokens=summary_budget_tokens,
        tokenizer=tokenizer,
    )
    transcript_text = summarize_with_budget(
        transcript_lines,
        budget_tokens=transcript_budget_tokens,
        tokenizer=tokenizer,
    )
    return summary_text, transcript_text

def generate_event_id(prefix: str = "rag") -> str:
    return f"{prefix}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def build_event_record(
    *,
    chat_id: str,
    event_id: str,
    query: str,
    response: str,
    event_hits: List[Dict[str, Any]],
    episode_hits: Optional[List[Dict[str, Any]]] = None,
    timestamp: datetime,
    time_window: Optional[TimeWindow],
    action_taken: Optional[str] = None,
    evidence_taken: Optional[str] = None,
    action_suggested: Optional[str] = None,
    evidence_suggested: Optional[str] = None,
) -> Dict[str, Any]:
    keywords = extract_keywords(f"{query} {response}")
    metadata: Dict[str, Any] = {
        "chat_id": chat_id,
        "retrieval": {
            "events": [
                {"event_id": hit.get("event_id"), "timestamp": hit.get("timestamp")}
                for hit in event_hits
            ]
        },
    }
    if episode_hits:
        metadata["retrieval"]["episodes"] = [
            {
                "episode_id": hit.get("episode_id"),
                "score": hit.get("score"),
                "month": hit.get("month"),
                "event_count": len(hit.get("events", [])),
                "episode_path": hit.get("episode_path"),
            }
            for hit in episode_hits
        ]
    if time_window:
        metadata["retrieval"]["time_window"] = time_window.as_dict()
    return {
        "event_id": event_id,
        "thread_id": chat_id,
        "timestamp": timestamp.isoformat(),
        "ts_unix": timestamp.timestamp(),
        "question": query.strip(),
        "response": response.strip(),
        "keywords": keywords,
        "action_taken": action_taken,
        "evidence_taken": evidence_taken,
        "action_suggested": action_suggested,
        "evidence_suggested": evidence_suggested,
        "source_suggested": "rag_agent",
        "metadata": metadata,
        "source_batch": "rag_agent_realtime",
        "batch_request_id": None,
        "raw_custom_id": event_id,
    }


def _default_should_use_memory(
    query: str,
    window: Optional[TimeWindow],
    hits: List[Dict[str, Any]],
    threshold: float,
) -> bool:
    lowered = query.lower()
    memory_hints = [
        "remember", "recall", "what did i", "what did we", "what did you",
        "earlier", "previous", "last time", "yesterday", "today", "last week",
        "last month", "last year", "ago",
    ]
    if window is not None:
        return True
    if any(hint in lowered for hint in memory_hints):
        return True
    if not hits:
        return False
    top_score = hits[0].get("score", 0.0)
    return top_score >= threshold


def build_chat_prompt(messages: List[Dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def generate_text(
    messages: List[Dict[str, str]],
    *,
    tokenizer: Any,
    model: Any,
    stream: bool = False,
    max_new_tokens: int = 400,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
) -> str:
    import threading

    import torch
    from transformers import TextIteratorStreamer

    prompt = build_chat_prompt(messages, tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    top_k = None
    if not do_sample:
        temperature = 1.0
        top_p = 1.0
        top_k = 0
    if not stream:
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = generated[:, encoded["input_ids"].shape[-1]:]
        return tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    chunks: List[str] = []
    for text in streamer:
        print(text, end="", flush=True)
        chunks.append(text)
    thread.join()
    return "".join(chunks).strip()


def extract_action_evidence(
    query: str,
    response: str,
    *,
    tokenizer: Any,
    model: Any,
    extract_max_new_tokens: int,
) -> Dict[str, Optional[str]]:
    action_taken = None
    evidence_taken = None
    sentence = select_action_sentence(query)
    if sentence:
        evidence_taken = sentence
        summary_messages = build_action_taken_messages(sentence)
        summary = generate_text(
            summary_messages,
            tokenizer=tokenizer,
            model=model,
            stream=False,
            max_new_tokens=extract_max_new_tokens,
            do_sample=False,
        )
        action_taken = parse_action_taken(summary)

    suggest_messages = build_action_suggest_messages(response)
    suggest = generate_text(
        suggest_messages,
        tokenizer=tokenizer,
        model=model,
        stream=False,
        max_new_tokens=extract_max_new_tokens,
        do_sample=False,
    )
    suggest_payload = parse_action_suggest(suggest)
    return {
        "action_taken": action_taken,
        "evidence_taken": evidence_taken,
        "action_suggested": suggest_payload.get("action_suggested"),
        "evidence_suggested": suggest_payload.get("evidence_suggested"),
    }


def run_chat_loop(
    *,
    model: Any,
    tokenizer: Any,
    storage: Any,
    chat_id: str,
    top_k: int,
    max_new_tokens: int,
    extract_max_new_tokens: int,
    event_context_budget_tokens: int,
    temperature: float,
    top_p: float,
    refresh_every: int,
    episode_lookback_months: int,
    episode_events_per_context: int,
    episode_top_k: int,
    episode_prefilter: bool,
    memory_score_threshold: float,
    allow_general_fallback: bool,
    memory_system_prompt: str,
    general_system_prompt: str,
    stream: bool = False,
    echo_input: bool = True,
    should_use_memory_fn: Optional[
        Callable[[str, Optional[TimeWindow], List[Dict[str, Any]], float], bool]
    ] = None,
) -> None:
    history: List[Dict[str, str]] = []
    query_count = 0
    last_event_hits: List[Dict[str, Any]] = []
    last_window: Optional[TimeWindow] = None
    if should_use_memory_fn is None:
        should_use_memory_fn = _default_should_use_memory

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)
        window = detect_time_request(query, now=timestamp)
        time_range = None
        if window is not None:
            time_range = (window.start.timestamp(), window.end.timestamp())

        episode_hits_full: List[Dict[str, Any]] = []
        episode_context_hits: List[Dict[str, Any]] = []
        allowed_event_ids = None
        if window is not None:
            episode_hits_full = storage.query_events_by_time(
                window.start,
                window.end,
                limit_episodes=episode_top_k,
                events_per_episode=None,
            )
        elif episode_prefilter:
            episode_hits_full = storage.retrieve_episode_context(
                query,
                limit=episode_top_k,
                months=episode_lookback_months,
                events_per_episode=None,
            )
        if episode_hits_full:
            allowed_event_ids = {
                evt.get("event_id")
                for hit in episode_hits_full
                for evt in hit.get("events", [])
                if evt.get("event_id")
            }
            episode_context_hits = trim_episode_hits(
                episode_hits_full,
                events_per_episode=episode_events_per_context,
            )
        lowered = query.lower()
        follow_up_hints = [
            "which date", "what date", "exact date", "which day", "what day",
            "that", "this", "it", "the error", "the answer", "that question",
        ]
        is_follow_up = (
            window is None
            and last_event_hits
            and (len(lowered.split()) <= 6 or any(hint in lowered for hint in follow_up_hints))
        )

        if is_follow_up:
            event_hits = list(last_event_hits)
            if last_window is not None:
                window = last_window
                time_range = (window.start.timestamp(), window.end.timestamp())
        else:
            event_hits = storage.retrieve(
                query,
                limit=top_k,
                time_window=time_range,
                allowed_event_ids=allowed_event_ids,
            )
        use_memory = should_use_memory_fn(query, window, event_hits, memory_score_threshold)
        episode_summary = "- None"
        episode_transcripts = ""
        if episode_context_hits:
            episode_summary, episode_transcripts = summarize_episode_hits(
                episode_context_hits,
                window=window,
                tokenizer=tokenizer,
            )
        event_context = (
            summarize_events(event_hits, budget_tokens=event_context_budget_tokens, tokenizer=tokenizer)
            if use_memory
            else "- None"
        )
        window_note = ""
        if window is not None:
            window_note = f"\nTime window: {window.label} ({window.start.isoformat()} -> {window.end.isoformat()})\n"

        if use_memory:
            blocks = [f"Relevant past events:\n{event_context}"]
            if episode_context_hits:
                blocks.append(f"Relevant episodes:\n{episode_summary}")
                if episode_transcripts.strip():
                    blocks.append(f"Episode excerpts:\n{episode_transcripts}")
            if window_note:
                blocks.append(window_note.strip())
            blocks.append(f"User question: {query}")
            context = "\n\n".join(blocks)
        else:
            context = f"User question: {query}"

        system_prompt = memory_system_prompt if (use_memory or not allow_general_fallback) else general_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": context},
        ]

        if echo_input:
            print(f"You: {query}")
        if stream:
            print("Assistant: ", end="", flush=True)
        response = generate_text(
            messages,
            tokenizer=tokenizer,
            model=model,
            stream=stream,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        if stream:
            print("")
        else:
            print(f"Assistant: {response}")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})
        history = history[-6:]

        event_id = generate_event_id("rag")
        action_payload = extract_action_evidence(
            query,
            response,
            tokenizer=tokenizer,
            model=model,
            extract_max_new_tokens=extract_max_new_tokens,
        )
        event_record = build_event_record(
            chat_id=chat_id,
            event_id=event_id,
            query=query,
            response=response,
            event_hits=event_hits,
            episode_hits=episode_context_hits,
            timestamp=timestamp,
            time_window=window,
            action_taken=action_payload.get("action_taken"),
            evidence_taken=action_payload.get("evidence_taken"),
            action_suggested=action_payload.get("action_suggested"),
            evidence_suggested=action_payload.get("evidence_suggested"),
        )
        storage.append_event(event_record)
        query_count += 1
        if use_memory and event_hits:
            last_event_hits = event_hits
            last_window = window

        if refresh_every > 0 and query_count % refresh_every == 0:
            storage.ensure_retrieval_index(refresh=True)
