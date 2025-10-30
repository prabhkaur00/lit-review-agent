# Literature Agent

Framework to generate quantitative QA pairs from local PDFs and evaluate a model against the verified dataset.

## Prereqs
- Python 3.9+
- `export OPENAI_API_KEY=...`
- `pip install openai python-docx`
---

## Step 1 — Generate QA pairs
**Script:** `generate_qna.py`  
Input:
A local pdf file.

Outputs:
- JSON: `{"papers":[...], "items":[{id,question,answer}, ...]}`
- DOCX: first non-empty line is `Papers: /abs/a.pdf, /abs/b.pdf`

**Single PDF**

The following command generates a .json file with QA pairs.
```bash
python generate_qna.py \
  --pdf /path/to/paper.pdf \
```

If you want a docx file along with the json (for easier editing), add --docx flag.
```bash
python generate_qna.py \
  --pdf /path/to/paper.pdf \
  --docx
```

**Multiple PDFs**
```bash
python generate_qna.py \
  --pdf /path/a.pdf /path/b.pdf \
  --docx
```


**Optional overrides:**.
```bash
--prompt prompts/prompt_qa_gen.txt \
--prompt_multi prompts/prompt_qa_gen_multi.txt
```

## Step 2 — Expert verification

Open the generated *_QA.docx and review/edit the QAs.

Please do not change the schema in case of the JSON, and header or labels ("Question:", "Answer:", "Papers:") in case of the docx.
These strings are parsed verbatim during evaluation.

## Step 3 — Evaluate against the verified dataset

Script: evaluate.py

This script fetches model answers for the verified QAs, and gives overall accuracy. Accepts JSON or DOCX via --data. If --pdf is omitted, PDFs are inferred from the dataset.

Input:
The verified json or docx.

Output:
A results file with model answers and overall accuracy.

Evaluate DOCX
```bash
python evaluate.py \
  --data /path/to/QA.docx \
 ```

Evaluate JSON
```bash
python evaluate.py \
  --data /path/to/QA.json \
``` 

If externally generated json/docx, also specify pdf file(s):
```bash
python evaluate.py \
  --data /path/to/QA.json \
  --pdf /path/to/paper.pdf
```


Optionally override the below arguments if needed:
```bash
python evaluate.py \
  --data /path/to/dataset.{json|docx} \
  --pdf /path/to/paper1.pdf /path/to/paper2.pdf \
  --prompt_answer prompts/prompt_answer.txt \
  --prompt_answer_multi prompts/prompt_answer_multi.txt \
  --prompt_eval prompts/prompt_eval.txt \
  --model-answer gpt-5 \
  --model-judge gpt-5 \
 ```
