"""SCORE prompts, prefixes and parsing, shared by training (ray_trainer.py) and eval.py."""

import re

# =============================================================================
# SCORE constants (paper Sec. 3 and Appendix "Prompts and Prefixes")
# =============================================================================
# Sandbox error ids, see verl/utils/reward_score/feedback/code.py
ERR_PASS, ERR_RE, ERR_WA, ERR_TLE, ERR_CE = 0, 1, 2, 3, 4
# Shared failure type of the binary mode (a single blind-spot rate is tracked)
FAIL = -1

DIAG_Q_BINARY = (
    "Given your solution above, will it pass all hidden tests? "
    "Answer with a single letter: A (pass) or B (fail)."
)
DIAG_Q_FIVE_CLASS = (
    "Given your solution above, predict the execution outcome. Answer with a single letter: "
    "A (pass), B (compile error), C (runtime error), D (wrong answer), or E (time limit exceeded)."
)
# Option letter <-> sandbox outcome. Binary: A = pass, B = fail.
LETTER_TO_ERR = {"A": ERR_PASS, "B": ERR_CE, "C": ERR_RE, "D": ERR_WA, "E": ERR_TLE}
ERR_TO_LETTER = {v: k for k, v in LETTER_TO_ERR.items()}

ERR_NAME = {ERR_CE: "compile error", ERR_RE: "runtime error", ERR_WA: "wrong answer", ERR_TLE: "time limit exceeded"}
REPAIR_HINT = {
    ERR_CE: "Check syntax, imports, and undefined names.",
    ERR_RE: "Check edge cases, type mismatches, and index bounds.",
    ERR_WA: "Re-read the problem statement; check boundary conditions and off-by-one errors.",
    ERR_TLE: "Reduce the complexity; use more efficient data structures.",
}
PREFIX_BLIND_SPOT = "You predicted that this solution passes, but it failed with {err}. This is a blind spot. {hint}\n\n"
PREFIX_WRONG_TYPE = "You predicted {pred} but the actual error was {err}.\n\n"
# Feedback-conditioned teacher prompt x~ = [x; y; f] (Eq. enriched)
TEACHER_TEMPLATE = (
    "{problem}\n\n"
    "Your earlier attempt:\n\n{attempt}\n\n"
    "The following is feedback from your unsuccessful earlier attempt:\n\n{feedback}\n\n"
    "Correctly solve the original question."
)


def strip_thinking(text: str) -> str:
    """Final response with the <think>...</think> trace removed."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


def parse_diagnosis(text: str, binary: bool) -> int:
    """First option letter in the answer -> predicted outcome (0 = pass).

    Binary mode returns ERR_PASS or FAIL. An answer without a valid letter counts as "pass".
    """
    letters = "AB" if binary else "ABCDE"
    m = re.search(rf"\b([{letters}])\b", strip_thinking(text))
    if m is None or m.group(1) == "A":
        return ERR_PASS
    return FAIL if binary else LETTER_TO_ERR[m.group(1)]


def build_diagnosis_messages(tokenizer, problem: str, response: str, binary: bool, max_len: int = 512) -> list[dict]:
    """Diagnosis prompt x_diag = [x; y_sol; q] (paper Eq. diag-prompt), at most max_len tokens.

    y_sol is the final response with the thinking trace removed. If the prompt is too long, the
    solution is truncated first (keeping at least half of the budget), then the problem.
    Rendered with thinking disabled.
    """
    question = DIAG_Q_BINARY if binary else DIAG_Q_FIVE_CLASS

    def render(p, s):
        return [{"role": "user", "content": f"{p}\n\n{s}\n\n{question}"}]

    overhead = len(tokenizer.apply_chat_template(render("", ""), tokenize=True, return_dict=False,
                                              add_generation_prompt=True, enable_thinking=False))
    solution = strip_thinking(response)
    avail = max(0, max_len - overhead - 4)  # 4 tokens of slack for re-tokenisation at the cut points
    p_ids = tokenizer.encode(problem, add_special_tokens=False)
    s_ids = tokenizer.encode(solution, add_special_tokens=False)
    if len(p_ids) + len(s_ids) > avail:
        s_keep = max(avail - len(p_ids), min(len(s_ids), avail // 2))
        s_ids = s_ids[:s_keep]
        p_ids = p_ids[: max(0, avail - len(s_ids))]
        problem, solution = tokenizer.decode(p_ids), tokenizer.decode(s_ids)
    return render(problem, solution)
