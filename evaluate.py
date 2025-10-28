import os, json, argparse, re
from openai import OpenAI
from datetime import datetime
from pathlib import Path

out_dir = Path("out")
out_dir.mkdir(parents=True, exist_ok=True)

def load_json(p): return json.load(open(p, "r", encoding="utf-8"))
def load_text(p): return open(p, "r", encoding="utf-8").read()

def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pdf = client.files.create(
        file=open("papers/1.pdf", "rb"),
        purpose="assistants"
    )
    pdf2 = client.files.create(
        file=open("papers/2.pdf", "rb"),
        purpose="assistants"
    )
    pdf3 = client.files.create(
        file=open("papers/3.pdf", "rb"),
        purpose="assistants"
    )
    dataset = "data/dataset_smol.json"
    dataset_multi = "data/dataset_multipaper.json"
    promptfile_a = "prompts/prompt-answer.txt"
    promptfile_a_multi = "prompts/prompt-answer.txt"
    promptfile_e = "prompts/prompt-eval.txt"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = out_dir / f"results_{ts}.json"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt_answer = load_text(promptfile_a)
    prompt_answer_multi = load_text(promptfile_a_multi)
    prompt_eval = load_text(promptfile_e)
    data = load_json(dataset)
    data_multi = load_json(dataset_multi)

    correct_count = 0
    results = []

    for ex in data_multi:
        q = ex["question"]
        gold = ex["answer"]

        # resp = client.responses.create(
        #     model="gpt-4o-mini",
        #     instructions=prompt_answer,
        #     input=[{
        #         "role":"user",
        #         "content":[
        #             {"type":"input_text","text":f"{q}\n"},
        #             {"type":"input_file","file_id":pdf.id}
        #         ]
        #     }]
        # )

        resp = client.responses.create(
            model="gpt-4o-mini",
            instructions=prompt_answer_multi,
            input=[{
                "role":"user",
                "content":[
                    {"type":"input_text","text":f"{q}\n"},
                    {"type":"input_file","file_id":pdf.id},
                    {"type":"input_file","file_id":pdf2.id},
                    {"type":"input_file","file_id":pdf3.id}

                ]
            }]
        )
        pred = resp.output_text.strip()

        judge_resp = client.responses.create(
        model="gpt-4o-mini",
        instructions=prompt_eval, 
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text":
                 f"QUERY:\n{q}\nGOLD_ANSWER:\n{gold}\nMODEL_ANSWER:\n{pred}"}
                ]
            }]
        )

        judge_text = judge_resp.output_text.strip().lower()
        is_correct = judge_text == "true"

        correct_count += int(is_correct)
        results.append({
            "id": ex["id"],
            "question": q,
            "gold_answer": gold,
            "model_answer": pred,
            "correct": is_correct,
            "judge_raw": judge_text
        })

    summary = {
        "total": len(data_multi),
        "correct": correct_count,
        "accuracy": round(correct_count / max(1, len(data_multi)), 3)
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
