"""Run the evaluation cases and write evaluation/results.md.

    python evaluation/run_eval.py

Uses the real knowledge/ folder and (if GROQ_API_KEY is set) the real model, so the
output documents retrieved sources and observed behaviour. Re-run after adding documents.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import pipeline  # noqa: E402
from document_loader import load_chunks  # noqa: E402
from retriever import Retriever  # noqa: E402


def main():
    chunks, docs, problems = load_chunks(config.KNOWLEDGE_DIR)
    retriever = Retriever(chunks)
    cases = json.loads((ROOT / "evaluation" / "cases.json").read_text(encoding="utf-8"))
    api_key = config.get_api_key()

    lines = [
        "# Evaluation results",
        "",
        f"- Reference documents loaded: {len(docs)}",
        f"- Skipped documents: {problems or 'none'}",
        f"- Answer generation: {'enabled' if api_key else 'disabled (no GROQ_API_KEY)'}",
        "",
        "| # | Question | Expected | Observed kind | Retrieved sources | Output (first 300 chars) |",
        "|---|----------|----------|---------------|-------------------|--------------------------|",
    ]
    replies = {}
    for case in cases:
        p = {"age_value": 25, "age_unit": "years", "sex": "Prefer not to say",
             "duration": "1-3 days", "severity": 4, **case.get("profile", {})}
        profile = pipeline.Profile.from_dict(p)
        history = []
        if "history_from" in case:
            prev = replies[case["history_from"]]
            history = [
                {"role": "user", "content": prev["question"], "profile": profile.to_dict()},
                {"role": "assistant", "content": prev["reply"].content, "kind": prev["reply"].kind},
            ]
        reply = pipeline.answer_question(case["question"], profile, history, retriever, api_key=api_key)
        replies[case["id"]] = {"question": case["question"], "reply": reply}
        found = "; ".join(f"[{s['label']}] {s['title']} ({s['publisher']}, v{s['version']})"
                          for s in reply.sources) or "none"
        out = reply.content.replace("\n", " ").replace("|", "/")[:300]
        lines.append(f"| {case['id']} | {case['question']} | {case['expect']} | {reply.kind} | {found} | {out} |")

    (ROOT / "evaluation" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote evaluation/results.md")


if __name__ == "__main__":
    main()
