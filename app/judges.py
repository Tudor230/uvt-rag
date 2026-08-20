"""LLM judge shims for evaluation.

DeepSeek's API rejects the structured-output (``json_schema``) transport that
``open_rag_eval.models.llm_judges.OpenAIModel.parse`` uses, so we subclass it
and override only ``parse`` to use DeepSeek's native JSON mode
(``response_format={"type": "json_object"}``) with the pydantic schema pasted
into the prompt.
"""
import copy
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from diskcache import Cache
from pydantic import BaseModel, TypeAdapter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from open_rag_eval.models.llm_judges import OpenAIModel
from open_rag_eval.metrics.umbrela_metric import UMBRELAMetric, UMBRELAScore

from ragas.cache import DiskCacheBackend

logger = logging.getLogger(__name__)

DEFAULT_CACHE_SIZE_LIMIT = 50 * 1024**3  # 50 GB


class SizedDiskCacheBackend(DiskCacheBackend):
    """Ragas DiskCacheBackend with a configurable size limit."""

    def __init__(self, cache_dir: str = ".cache", size_limit: int = DEFAULT_CACHE_SIZE_LIMIT):
        from diskcache import Cache

        self.cache = Cache(cache_dir, size_limit=size_limit)


class DeepSeekJudge(OpenAIModel):
    """OpenAIModel shim that talks to DeepSeek's native JSON mode.

    ``parse`` reproduces ``OpenAIModel.parse`` behaviour (pydantic-validated
    structured output) but:

    * uses ``chat.completions.create`` with ``response_format={"type":
      "json_object"}`` — the only structured format DeepSeek accepts;
    * appends the JSON schema to the prompt (DeepSeek requires the word
      "json" for ``json_object`` mode);
    * strips kwargs DeepSeek does not document (``seed``) — the rest are
      ignored harmlessly;
    * disables thinking mode to cut latency/cost and remove variance.
    """

    _STRIP_KWARGS = ("seed",)

    def __init__(self, model_options: dict):
        self._api_key = model_options["api_key"]
        os.environ.setdefault("OPENAI_API_KEY", self._api_key)
        super().__init__(model_options)
        self.client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=model_options.get("timeout", 600.0),
            max_retries=model_options.get("max_retries", 5),
        )
        self.thinking = model_options.get("thinking")

        cache_dir = model_options.get("cache_dir")
        cache_size_limit = model_options.get(
            "cache_size_limit", DEFAULT_CACHE_SIZE_LIMIT
        )
        self._cache = (
            Cache(cache_dir, size_limit=cache_size_limit) if cache_dir else None
        )
        if self._cache is not None:
            logger.info(
                "DeepSeek judge cache enabled at %s (size_limit=%d bytes)",
                cache_dir,
                cache_size_limit,
            )

    def _cache_key(self, structured_prompt: str, response_format: BaseModel) -> str:
        payload = {
            "model": self.model_name,
            "schema": response_format.__name__,
            "prompt": structured_prompt,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _clean_kwargs(self, model_kwargs: dict | None) -> dict:
        kwargs = dict(model_kwargs or {})
        for key in self._STRIP_KWARGS:
            kwargs.pop(key, None)
        kwargs.setdefault("temperature", 0.0)
        return kwargs

    def _call(self, messages: list[dict], model_kwargs: dict | None) -> object:
        kwargs = self._clean_kwargs(model_kwargs)
        extra = {}
        if self.thinking:
            extra = {"thinking": {"type": self.thinking}}
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body=extra,
            **kwargs,
        )

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APIError,
                ValueError,
                json.JSONDecodeError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def parse(self, prompt: str, response_format: BaseModel, model_kwargs=None):
        schema = response_format.model_json_schema()

        structured_prompt = f"""{prompt}

Please respond with a JSON object that matches this exact schema:
{json.dumps(schema, indent=2)}

Return only the JSON object, no other text."""

        cache_key = None
        if self._cache is not None:
            cache_key = self._cache_key(structured_prompt, response_format)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("DeepSeek judge cache hit (%s)", response_format.__name__)
                return copy.deepcopy(cached)

        completion = self._call(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that follows user "
                    "instructions precisely and provides accurate information.",
                },
                {"role": "user", "content": structured_prompt},
            ],
            model_kwargs,
        )

        metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if getattr(completion, "usage", None):
            metadata["input_tokens"] = getattr(completion.usage, "prompt_tokens", 0)
            metadata["output_tokens"] = getattr(completion.usage, "completion_tokens", 0)
            metadata["total_tokens"] = getattr(completion.usage, "total_tokens", 0)
        else:
            logger.warning("Token usage not available from DeepSeek response")

        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise ValueError(
                f"DeepSeek returned empty content for schema parse. "
                f"Full completion: {completion}"
            )

        response_json = json.loads(content)
        self._coerce_enum_strs(response_json, schema)
        parsed_response = TypeAdapter(response_format).validate_python(response_json)

        result = {
            "response": parsed_response,
            "metadata": metadata,
        }
        if cache_key is not None:
            self._cache.set(cache_key, result)
        return result

    @staticmethod
    def _coerce_enum_strs(response_json: dict, schema: dict) -> None:
        """Coerce int values into str enums (DeepSeek returns bare ints).

        e.g. ``UMBRELAScoreValues`` declares string members ('0'..'3') but the
        model answers ``{"score": 2}`` with an integer. Convert in place so
        the pydantic enum validates.
        """
        defs = schema.get("$defs", {})

        def _resolve(prop: dict) -> dict:
            ref = prop.get("$ref")
            if ref and ref.startswith("#/$defs/"):
                return defs.get(ref.split("/")[-1], prop)
            return prop

        for field_name, prop in schema.get("properties", {}).items():
            resolved = _resolve(prop)
            enum = resolved.get("enum")
            if isinstance(enum, list) and all(isinstance(e, str) for e in enum):
                value = response_json.get(field_name)
                if isinstance(value, int) and str(value) in enum:
                    response_json[field_name] = str(value)


class ParallelUMBRELAMetric(UMBRELAMetric):
    """UMBRELAMetric that scores the passages of a query concurrently.

    The upstream ``compute`` loops over ``retrieved_passages`` sequentially,
    calling the judge once per passage — with a remote judge that serializes
    all the network latency. We replicate the loop (same prompt selection,
    same token accounting, same failure semantics) but run the per-passage
    judge calls in a thread pool.

    ``DeepSeekJudge.parse`` is synchronous and its OpenAI client is safe for
    concurrent use from multiple threads.
    """

    def __init__(self, model, max_workers: int = 8):
        super().__init__(model)
        self.max_workers = max_workers

    def _build_prompt(self, query: str, passage: str) -> str:
        if (
            "gpt-oss" in self.model.model_name.lower()
            or "qwen" in self.model.model_name.lower()
        ):
            return self._UMBRELA_NEW_PROMPT.format(query=query, passage=passage)
        return self._UMBRELA_PROMPT.format(query=query, passage=passage)

    def _score_passage(self, key: str, passage: str, query: str):
        prompt = self._build_prompt(query, passage)
        return self.model.parse(prompt, UMBRELAScore, self.model_kwargs)

    def compute(self, retrieval_result, k_values: list[int]) -> dict:
        scores = {"umbrela_scores": {}}
        umbrela_scores = scores["umbrela_scores"]

        total_input_tokens = 0
        total_output_tokens = 0

        passages = retrieval_result.retrieved_passages
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._score_passage, key, passage, retrieval_result.query
                ): key
                for key, passage in passages.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise Exception(
                        f"Error computing UMBRELA score: {str(exc)}"
                    ) from exc

                response = result["response"]
                metadata = result["metadata"]
                total_input_tokens += metadata.get("input_tokens", 0)
                total_output_tokens += metadata.get("output_tokens", 0)

                if not response.score:
                    raise ValueError(
                        f"Failed to parse response: {response.refusal}"
                    )
                umbrela_scores[key] = int(response.score.value)

        scores["token_usage"] = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

        self.add_retrieval_metrics(scores, k_values)
        return scores
