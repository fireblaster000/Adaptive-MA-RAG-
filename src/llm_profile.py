import time
from typing import Any, Callable, Dict, Optional, Tuple

from langchain_community.callbacks import get_openai_callback


def profile_llm_call(
    fn: Callable[[], Any],
    *,
    stage: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Profile a single LLM call using LangChain's OpenAI callback.

    Returns:
      (result, metric_dict)
    """
    start = time.perf_counter()
    with get_openai_callback() as cb:
        result = fn()
    latency_ms = (time.perf_counter() - start) * 1000.0

    metric: Dict[str, Any] = {
        "stage": stage,
        "latency_ms": latency_ms,
        "prompt_tokens": getattr(cb, "prompt_tokens", None),
        "completion_tokens": getattr(cb, "completion_tokens", None),
        "total_tokens": getattr(cb, "total_tokens", None),
        "total_cost": getattr(cb, "total_cost", None),
    }
    if extra:
        metric["extra"] = extra
    return result, metric

