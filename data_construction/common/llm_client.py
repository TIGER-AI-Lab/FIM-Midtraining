#!/usr/bin/env python3
"""
Shared Gemini client and token/cost accounting for the step-4 pipelines.

Both the single-function and multi-function step 4 issue the same two kinds of
call ("fim" and "critique"), so the client, the retry policy and the cost
counter live here rather than being duplicated on both sides.
"""

import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional W&B integration — the pipelines run fine without it.
try:
    import wandb
    import weave
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    weave = None
    WANDB_AVAILABLE = False


# The env var the google-genai SDK itself prefers, then the historical fallback.
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def resolve_api_key() -> Tuple[str, str]:
    """Return (api_key, env_var_name) or exit with an actionable message."""
    for var in API_KEY_ENV_VARS:
        key = os.getenv(var)
        if key:
            return key, var
    raise SystemExit(
        "No Gemini API key found. Set one of "
        f"{' or '.join(API_KEY_ENV_VARS)}:\n"
        "    export GEMINI_API_KEY='your-key-here'\n"
        "Get a key at https://aistudio.google.com/apikey"
    )


class TokenCounter:
    """Track token usage and running cost. Safe to share across worker threads."""

    def __init__(
        self,
        price_per_1m_input_tokens: float = 0.50,
        price_per_1m_output_tokens: float = 3.00,
    ):
        self.price_in = price_per_1m_input_tokens
        self.price_out = price_per_1m_output_tokens
        self._lock = threading.Lock()

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        self.fim_input_tokens = 0
        self.fim_output_tokens = 0
        self.fim_requests = 0
        self.critique_input_tokens = 0
        self.critique_output_tokens = 0
        self.critique_requests = 0

    def add_usage(self, input_tokens: int, output_tokens: int, call_type: str = "fim"):
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.request_count += 1
            if call_type == "fim":
                self.fim_input_tokens += input_tokens
                self.fim_output_tokens += output_tokens
                self.fim_requests += 1
            elif call_type == "critique":
                self.critique_input_tokens += input_tokens
                self.critique_output_tokens += output_tokens
                self.critique_requests += 1

    def get_cost(self) -> dict:
        input_cost = (self.total_input_tokens / 1_000_000) * self.price_in
        output_cost = (self.total_output_tokens / 1_000_000) * self.price_out
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'request_count': self.request_count,
            'fim_requests': self.fim_requests,
            'critique_requests': self.critique_requests,
        }

    def print_cost_summary(self):
        cost = self.get_cost()
        print(f"\n💰 Token Usage & Cost (estimate):")
        print(f"   Input tokens:  {cost['input_tokens']:,} (${cost['input_cost']:.4f})")
        print(f"   Output tokens: {cost['output_tokens']:,} (${cost['output_cost']:.4f})")
        print(f"   Total tokens:  {cost['total_tokens']:,}")
        print(f"   FIM requests:  {cost['fim_requests']}")
        print(f"   Critique requests: {cost['critique_requests']}")
        print(f"   💵 Total cost: ${cost['total_cost']:.4f}")

    def to_dict(self) -> dict:
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'request_count': self.request_count,
            'fim_input_tokens': self.fim_input_tokens,
            'fim_output_tokens': self.fim_output_tokens,
            'fim_requests': self.fim_requests,
            'critique_input_tokens': self.critique_input_tokens,
            'critique_output_tokens': self.critique_output_tokens,
            'critique_requests': self.critique_requests,
        }

    def load_from_dict(self, data: dict):
        for key in (
            'total_input_tokens', 'total_output_tokens', 'request_count',
            'fim_input_tokens', 'fim_output_tokens', 'fim_requests',
            'critique_input_tokens', 'critique_output_tokens', 'critique_requests',
        ):
            setattr(self, key, data.get(key, 0))


class GeminiClient:
    """Gemini API client for code completion and critique, with retry + backoff."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        token_counter: Optional[TokenCounter] = None,
        use_wandb: bool = False,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        try:
            from google import genai
        except ImportError as exc:
            raise SystemExit(
                "The google-genai SDK is required for step 4. Install it with:\n"
                "    pip install google-genai"
            ) from exc

        api_key, env_var = resolve_api_key()
        logger.info(f"🔑 Using Gemini API key from ${env_var}")

        # Pass the key explicitly rather than letting the SDK pick an env var —
        # otherwise setting only GOOGLE_API_KEY silently authenticates with a
        # different variable than the one we validated.
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.token_counter = token_counter
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_response(
        self,
        prompt: str,
        sample_id: Optional[str] = None,
        call_type: str = "fim",
        temperature: float = 0.7,
    ) -> Tuple[str, dict]:
        """Send one prompt. Returns (response_text, usage_info). Raises on final failure."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": temperature, "top_p": 0.95},
                )
                latency = time.time() - start_time

                usage_info = self._extract_usage(response, call_type)
                usage_info['latency'] = latency

                if self.use_wandb:
                    self._log_to_wandb(sample_id, call_type, usage_info, latency, temperature)

                return response.text, usage_info

            except Exception as e:
                last_error = e
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

        logger.error(f"Gemini API call failed after {self.max_retries} attempts: {last_error}")
        raise last_error

    def _extract_usage(self, response, call_type: str) -> dict:
        usage_info = {'input_tokens': 0, 'output_tokens': 0}
        usage = getattr(response, 'usage_metadata', None)
        if usage is not None:
            usage_info['input_tokens'] = getattr(usage, 'prompt_token_count', 0) or 0
            usage_info['output_tokens'] = getattr(usage, 'candidates_token_count', 0) or 0
            if self.token_counter:
                self.token_counter.add_usage(
                    usage_info['input_tokens'],
                    usage_info['output_tokens'],
                    call_type,
                )
        return usage_info

    def _log_to_wandb(self, sample_id, call_type, usage_info, latency, temperature):
        price_in = self.token_counter.price_in if self.token_counter else 0.0
        price_out = self.token_counter.price_out if self.token_counter else 0.0
        cost = (
            (usage_info['input_tokens'] / 1_000_000) * price_in
            + (usage_info['output_tokens'] / 1_000_000) * price_out
        )
        wandb.log({
            f"api_call/{call_type}_sample_id": sample_id,
            f"api_call/{call_type}_input_tokens": usage_info['input_tokens'],
            f"api_call/{call_type}_output_tokens": usage_info['output_tokens'],
            f"api_call/{call_type}_cost": cost,
            f"api_call/{call_type}_latency": latency,
            f"api_call/{call_type}_temperature": temperature,
        })


def init_wandb(
    project: str,
    run_name: str,
    entity: str = "",
    config: Optional[Dict] = None,
) -> bool:
    """Initialise W&B + Weave. Returns False (with a warning) if not installed."""
    if not WANDB_AVAILABLE:
        logger.warning("⚠️ W&B requested but wandb/weave are not installed — continuing without it.")
        return False

    wandb.init(
        project=project,
        entity=entity or None,
        name=run_name,
        config=config or {},
        resume="allow",
    )
    weave.init(project)
    logger.info(f"✅ W&B initialized: {wandb.run.url}")
    return True


def finish_wandb():
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
