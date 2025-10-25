from openai import OpenAI
import os
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pdf = client.files.create(
    file=open("papers/DeepSeek_OCR_paper.pdf", "rb"),
    purpose="assistants"
)

resp = client.responses.create(
    model="gpt-4o-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Summarize section 2 and list the main contributions."},
            {"type": "input_file", "file_id": pdf.id}
        ]
    }]
)

print(resp.output_text)
