"""Reward function used by SCORE (custom_reward_function.path points here).

Binary reward: 1 if the program passes every test of the problem, else 0 (paper Sec. 2).
The result dict also carries the outcome type (error_type) and the textual feedback.
"""
from verl.utils.reward_score.feedback import code


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    if data_source not in ("code", "livecodebench", "humanevalplus"):
        raise ValueError(f"Unsupported data_source for SCORE: {data_source}")
    return code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
