# Lit-Review Agent

Framework to generate quantitative QA pairs from local PDFs and evaluate a model against the verified dataset.

## Prereqs
- Python 3.9+
- `export OPENAI_API_KEY=...`
- `pip install openai python-docx`
---

## Step 1 — Generate QA pairs
**Script:** `generate_qna.py`  
Outputs:
- JSON: `{"papers":[...], "items":[{id,question,answer}, ...]}`
- DOCX: first non-empty line is `Papers: /abs/a.pdf, /abs/b.pdf`

**Single PDF**
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


**Optional prompt overrides:**
```bash
--prompt prompts/prompt_qa_gen.txt \
--prompt_multi prompts/prompt_qa_gen_multi.txt
```

## Step 2 — Expert verification

Open the generated *_QA.docx and clean/fix QAs.
Keep the first line intact:

Papers: /abs/a.pdf, /abs/b.pdf

## Step 3 — Evaluate against the verified dataset

Script: evaluate.py

Accepts JSON or DOCX via --data. If --pdf is omitted, PDFs are inferred from the dataset.

Evaluate DOCX
```bash
python evaluate.py \
  --data out/multi_2_papers_QA.docx \
 ```


Evaluate JSON
```bash
python evaluate.py \
  --data out/multi_2_papers_QA.json \
``` 


Optionally override PDFs
```bash
python evaluate.py \
  --dataset out/multi_2_papers_QA.json \
  --pdf /path/a.pdf /path/b.pdf \
 ```
