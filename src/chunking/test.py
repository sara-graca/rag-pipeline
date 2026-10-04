from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PERSIST_DIR = "./chroma_db_rulebook"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# 1. Did everything get stored?
print("Chunks in database:", db._collection.count())

# 2. Does retrieval return sensible results?
for question in [
    "What is the rule for grenades?",
    "How can I use a stratagem?",
    "What are the different ways to score points?",
]:
    print(f"\n=== {question}")
    for doc in db.similarity_search(question, k=3):
        print("-", doc.metadata)
        print("  ", doc.page_content.replace("\n", " "))
