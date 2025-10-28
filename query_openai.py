from openai import OpenAI
import os
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pdf = client.files.create(
    file=open("/Users/prabhleenkaur/Desktop/FA25/PICASSO/code/lit-review-agent/sample_paper.pdf", "rb"),
    purpose="assistants"
)

pdf2 = client.files.create(
    file=open("/Users/prabhleenkaur/Desktop/FA25/PICASSO/code/lit-review-agent/papers/DeepSeek_OCR_paper.pdf", "rb"),
    purpose="assistants"
)

resp = client.responses.create(
    model="gpt-4o-mini",
    input=[{
        "role": "user",
        "content": [
            # {"type": "input_text", "text": "List figures with page numbers/captions you detect"},
            # {"type": "input_text", "text":
            #   "Can you explain the trend that the green curve followsi in fig. 4d, at what values it increases/decreases/maxima/minima?"},
            # {"type": "input_text", "text": 
            #  "What is the value q embedded in Fig. 4g?"},
            {"type": "input_text", "text": 
             "What are the similarities in the two papers, in detail?"},
            # {"type": "input_text", "text": "What does Fig. 4 show?"},
            # {"type": "input_text", "text": "What is the edge-state decay length ξ extracted from the dI/dV intensity profile in Fig. 4e, and how does it compare with the theoretical edge-state localization length in Fig. 4a–b?"},
            {"type": "input_file", "file_id": pdf.id},
            {"type": "input_file", "file_id": pdf2.id},
        ]
    }]
)

print(resp.output_text)
