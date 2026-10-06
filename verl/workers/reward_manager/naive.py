# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Force unbuffered stdout in Ray worker processes
if not os.environ.get("PYTHONUNBUFFERED"):
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Number of parallel workers for reward computation.
_REWARD_WORKERS = int(os.environ.get("VERL_REWARD_WORKERS", 8))

# How to isolate the reward sandbox from this (Ray) process:
#   subprocess (default) - run compute_score in a persistent pool of plain
#                          subprocesses (sandbox_pool.py); the per-test fork()
#                          in feedback/code.py then happens in Ray-free
#                          children. Fixes the Raylet/Plasma crashes
#                          ("Invalid Plasma message type") seen when the
#                          sandbox forked directly out of the Ray worker.
#   thread               - legacy in-process ThreadPool path (the crashy one);
#                          kept for A/B comparison.
_SANDBOX_ISOLATION = os.environ.get("VERL_SANDBOX_ISOLATION", "subprocess").lower()
# Per-item wall clock budget inside a sandbox worker (code.py's own per-test
# timeouts normally finish far sooner; this is the safety net for a hang).
_SANDBOX_TASK_TIMEOUT = float(os.environ.get("VERL_SANDBOX_TASK_TIMEOUT", 300))


def _sandbox_target(compute_score):
    """(module_path, fn_name, reward_kwargs) so spawn workers can re-import
    the reward fn, or None if it can't be resolved (-> thread fallback).

    verl's get_custom_reward_fn wraps the raw fn as
    partial(_call_with_kwargs, raw_fn, reward_kwargs); unwrap that shape.
    """
    fn = compute_score
    reward_kwargs = {}
    if isinstance(fn, functools.partial):
        callables = [a for a in fn.args if callable(a)]
        dicts = [a for a in fn.args if isinstance(a, dict)]
        if callables:
            fn = callables[0]
        if dicts:
            reward_kwargs = dicts[0]
    try:
        path = inspect.getsourcefile(fn)
        name = fn.__name__
    except Exception:
        return None
    if not path or not os.path.isfile(path) or name == "<lambda>":
        return None
    return path, name, reward_kwargs


def _failed_result(extra_info: dict) -> dict:
    """Score dict for an item whose sandbox worker died or timed out (same keys as feedback/code.py)."""
    from verl.utils.reward_score.feedback.code import ERROR_TYPE_TIMEOUT, ERROR_TYPE_LABELS

    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "incorrect_format": 0,
        "error_in_test_cases": 0,
        "timed_out": 1,
        "truncated": 1 if extra_info.get("truncated", False) else 0,
        "truncated_and_missing_answer": 0,
        "error_type": ERROR_TYPE_TIMEOUT,
        "error_type_label": ERROR_TYPE_LABELS[ERROR_TYPE_TIMEOUT],
        "feedback": "Time Limit Exceeded",
    }


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # Resolve (path, name, kwargs) once so sandbox workers can re-import the fn.
        self._sandbox_target = _sandbox_target(self.compute_score)
        if _SANDBOX_ISOLATION == "subprocess" and self._sandbox_target is None:
            print(
                "[Sandbox] WARNING: could not resolve compute_score to a file/name; "
                "falling back to the in-process thread path (Raylet-crash prone).",
                flush=True,
            )

    def __call__(self, data: DataProto, return_dict: bool = True) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Pre-extract all inputs (tokenizer decode must happen in main thread)
        items = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            # copy: the extra_info dict is shared by the N rollouts of a prompt
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", None) or {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_info["truncated"] = not (valid_response_ids == self.tokenizer.eos_token_id).any().item()

            items.append({
                "idx": i,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
                "valid_response_length": valid_response_length,
            })

        sandbox_items = items

        # Score function
        def _score_one(item):
            return item["idx"], self.compute_score(
                data_source=item["data_source"],
                solution_str=item["response_str"],
                ground_truth=item["ground_truth"],
                extra_info=item["extra_info"],
            )

        scores = [None] * len(data)

        # Score sandbox items. Default: persistent pool of plain subprocesses
        # (never multiprocessing fork/spawn of THIS Ray process), so the
        # per-test fork() in feedback/code.py happens in clean Ray-free
        # children -- see sandbox_pool.py for the crash this avoids.
        from tqdm import tqdm
        if sandbox_items and _SANDBOX_ISOLATION == "subprocess" and self._sandbox_target is not None:
            from verl.workers.reward_manager import sandbox_pool

            path, name, rkw = self._sandbox_target
            pool = sandbox_pool.get_pool(path, name, _REWARD_WORKERS, rkw)
            batch = [
                {
                    "data_source": item["data_source"],
                    "solution_str": item["response_str"],
                    "ground_truth": item["ground_truth"],
                    "extra_info": item["extra_info"],
                }
                for item in sandbox_items
            ]
            results = pool.score_batch(
                batch, timeout=_SANDBOX_TASK_TIMEOUT, progress_desc="[Sandbox] compute_score (subproc)"
            )
            failed = 0
            for item, result in zip(sandbox_items, results):
                if result is None:  # worker died / timed out / fn raised
                    failed += 1
                    result = _failed_result(item["extra_info"])
                scores[item["idx"]] = result
            if failed:
                print(f"[Sandbox] {failed}/{len(sandbox_items)} item(s) scored 0 after worker failure.", flush=True)
        elif sandbox_items:
            # Legacy in-process thread path (VERL_SANDBOX_ISOLATION=thread).
            with ThreadPoolExecutor(max_workers=_REWARD_WORKERS) as pool:
                futures = {pool.submit(_score_one, item): item for item in sandbox_items}
                pbar = tqdm(as_completed(futures), total=len(sandbox_items), desc="[Sandbox] compute_score", leave=False)
                for future in pbar:
                    idx, score = future.result()
                    scores[idx] = score
                pbar.close()

        # Collect results
        already_print_data_sources = {}
        for i, item in enumerate(items):
            score = scores[item["idx"]]
            valid_response_length = item["valid_response_length"]
            data_source = item["data_source"]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[item["idx"], valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", item["prompt_str"])
                print("[response]", item["response_str"])
                print("[ground_truth]", item["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
