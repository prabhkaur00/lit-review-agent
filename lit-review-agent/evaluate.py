import os, json, argparse
from openai import OpenAI
from datetime import datetime
from pathlib import Path

from qna_parse import load_pairs_from_path, infer_pdfs_from_dataset

DEF_PROMPT_ANSWER = "prompts/prompt_answer.txt"
DEF_PROMPT_ANSWER_MULTI = "prompts/prompt_answer_multi.txt"
DEF_PROMPT_EVAL = "prompts/prompt_eval.txt"

def load_text(p): return open(p, "r", encoding="utf-8").read()

def run(
    pdf_paths,
    dataset_path,
    prompt_answer_path=None,
    prompt_answer_multi_path=None,
    prompt_eval_path=None,
    model_answer="gpt-5",
    model_judge="gpt-5",
    out_path=None,
    out_dir="out",
    timestamped=True,
):
    # Normalize/resolve PDFs; allow omission (fallback to dataset)
    if isinstance(pdf_paths, (str, Path)):
        pdf_paths = [str(pdf_paths)]
    pdf_paths = [str(p) for p in (pdf_paths or [])]

    if not pdf_paths:
        pdf_paths = infer_pdfs_from_dataset(dataset_path)
        if not pdf_paths:
            raise RuntimeError(
                "No PDFs provided via --pdf, and dataset does not specify papers.\n"
                "Expected either:\n"
                "  - JSON with top-level key: {\"papers\": [\"/abs/a.pdf\", ...]}\n"
                "  - DOCX whose first non-empty line starts with: 'Papers: /abs/a.pdf, /abs/b.pdf'"
            )

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(out_path) if out_path else out_dir / f"results_{ts if timestamped else 'latest'}.json"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Upload PDFs (close file handles promptly)
    uploaded = []
    for p in pdf_paths:
        with open(p, "rb") as fh:
            uploaded.append(client.files.create(file=fh, purpose="assistants"))

    # Choose prompt based on count
    use_multi = len(uploaded) > 1
    if use_multi:
        chosen_prompt_answer_path = (prompt_answer_multi_path or prompt_answer_path or DEF_PROMPT_ANSWER_MULTI)
    else:
        chosen_prompt_answer_path = (prompt_answer_path or DEF_PROMPT_ANSWER)

    prompt_eval_path = prompt_eval_path or DEF_PROMPT_EVAL
    prompt_answer = load_text(chosen_prompt_answer_path)
    prompt_eval = load_text(prompt_eval_path)

    dataset_items = load_pairs_from_path(dataset_path)
    if not dataset_items:
        raise RuntimeError(f"No QA pairs found in dataset: {dataset_path}")

    correct_count, results = 0, []

    for ex in dataset_items:
        q, gold = ex["question"], ex["answer"]

        content = [{"type": "input_text", "text": f"{q}\n"}]
        for f in uploaded:
            content.append({"type": "input_file", "file_id": f.id})

        resp = client.responses.create(
            model=model_answer,
            instructions=prompt_answer,
            input=[{"role": "user", "content": content}],
        )
        pred = (resp.output_text or "").strip()

        judge_resp = client.responses.create(
            model=model_judge,
            instructions=prompt_eval,
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": f"QUERY:\n{q}\nGOLD_ANSWER:\n{gold}\nMODEL_ANSWER:\n{pred}"}],
            }],
        )
        judge_text = (judge_resp.output_text or "").strip().lower()
        is_correct = judge_text == "true"

        correct_count += int(is_correct)
        results.append({
            "id": ex.get("id"),
            "question": q,
            "gold_answer": gold,
            "model_answer": pred,
            "correct": is_correct,
            "judge_raw": judge_text,
        })

    summary = {
        "total": len(dataset_items),
        "correct": correct_count,
        "accuracy": round(correct_count / max(1, len(dataset_items)), 3),
        "papers": [str(Path(p).resolve()) for p in pdf_paths],  # PDFs actually used
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", dest="pdfs", nargs="+", required=False,
                    help="Optional: one or more PDF paths. If omitted, the script will read 'papers' from the dataset JSON or the 'Papers:' line in the DOCX.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt_answer", default=None)
    ap.add_argument("--prompt_answer_multi", default=None)
    ap.add_argument("--prompt_eval", default=None)
    ap.add_argument("--model-answer", dest="model_answer", default="gpt-5")
    ap.add_argument("--model-judge", dest="model_judge", default="gpt-5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--no-timestamp", action="store_true")
    args = ap.parse_args()

    run(
        pdf_paths=args.pdfs,
        dataset_path=args.dataset,
        prompt_answer_path=args.prompt_answer,
        prompt_answer_multi_path=args.prompt_answer_multi,
        prompt_eval_path=args.prompt_eval,
        model_answer=args.model_answer,
        model_judge=args.model_judge,
        out_path=args.out,
        out_dir=args.out_dir,
        timestamped=not args.no_timestamp,
    )
