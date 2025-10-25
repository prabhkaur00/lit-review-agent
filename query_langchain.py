from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

pdf_paths = ["/Users/prabhleenkaur/Desktop/FA25/PICASSO/code/lit-review-agent/papers/DeepSeek_OCR_paper.pdf"]

raw = []
for p in pdf_paths:
    for d in PyPDFLoader(p).load():
        d.metadata["source"] = p
        raw.append(d)

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
splits = splitter.split_documents(raw)

emb = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

# emb = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=os.environ["OPENAI_API_KEY"],
# )

vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_kwargs={"k": 6})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                            google_api_key=os.environ["GGOLE_API_KEY"],
                            temperature=0)

# llm = ChatOpenAI(
#     model="gpt-4o-mini",        # or "gpt-4o", "gpt-4.1-mini", etc.
#     api_key=os.environ["OPENAI_API_KEY"],
# )

def format_doc(docs):
    out = []
    for d in docs:
        page = d.metadata.get("page")
        src = d.metadata.get("source", "source")
        tag = f"{src} p.{(page+1) if page is not None else '?'}"
        out.append(f"[{tag}] {d.page_content.strip()}")
    return "\n\n".join(out)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer strictly from Context. Use inline citations like [source:filename.pdf p.X]. If unsure, say so."),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations.")
])

chain = (
    RunnableParallel(
        context=retriever | format_doc,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)

def ask(q: str) -> str:
    return chain.invoke(q)

if __name__ == "__main__":
    print(ask("Summarize the main contributions in this paper."))
