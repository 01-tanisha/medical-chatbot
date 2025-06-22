from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import CTransformers
import os, sqlite3, json
from flask import Flask, request, render_template, redirect, url_for, jsonify
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from datetime import datetime, timedelta, timezone


# Function to format timestamps
def format_timestamp(timestamp_input):
    try:
        # Parse timestamp string
        if isinstance(timestamp_input, str):
            try:
                dt = datetime.strptime(timestamp_input, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                dt = datetime.strptime(timestamp_input, "%Y-%m-%d %H:%M:%S")
        elif isinstance(timestamp_input, datetime):
            dt = timestamp_input
        else:
            return "Unknown time"

        # Convert from UTC to local time (e.g., IST = UTC+5:30)
        utc_dt = dt.replace(tzinfo=timezone.utc)
        local_dt = utc_dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
        now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        if local_dt.date() == now.date():
            return f"Today at {local_dt.strftime('%I:%M %p')}"
        elif local_dt.date() == (now - timedelta(days=1)).date():
            return f"Yesterday at {local_dt.strftime('%I:%M %p')}"
        elif local_dt.date() == (now + timedelta(days=1)).date():
            return f"Tomorrow at {local_dt.strftime('%I:%M %p')}"
        else:
            return local_dt.strftime('%d %b at %I:%M %p')
    except Exception:
        return "Unknown time"


# Load environment variables and setup app
load_dotenv()
app = Flask(__name__)

#configs
PDF_PATH = "Medical_book.pdf" # Path to your PDF file
MODEL_PATH = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"


#Extract data from file
def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    except Exception as e:
        return []
extracted_data = load_pdf("C:/Medical chat bot/Medical_book.pdf")


#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks = text_split(extracted_data)


#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embeddings model loaded.")
    return embeddings
embeddings = download_hugging_face_embeddings()


#store embeddings in FAISS
FAISS_INDEX_PATH = "faiss_index"
if os.path.exists(f"{FAISS_INDEX_PATH}.faiss") and os.path.exists(f"{FAISS_INDEX_PATH}.pkl"):
    print("Loading existing FAISS index...")
    docsearch = FAISS.load_local(FAISS_INDEX_PATH, embeddings,allow_dangerous_deserialization=True)
else:
    print("Creating FAISS index from scratch...")
    docsearch = FAISS.from_documents(text_chunks, embedding=embeddings)
    docsearch.save_local(FAISS_INDEX_PATH)



#prompt setup
prompt_template = """
I'm medical assistant. Only use the context provided from the PDF document to answer the user's question.
Do not use any external knowledge. If the answer is not in the document, reply: "Sorry, I don’t have information on that in the provided document."
Context: {context}
Question: {question}
Helpful answer:
"""


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}


# Load LLaMA model
if not os.path.exists("C:/Medical chat bot/model/llama-2-7b-chat.ggmlv3.q4_0.bin"):
    exit()


#sensure the model file exists at the specified path
llm = CTransformers(
    model="C:/Medical chat bot/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 256, 'temperature': 0.3}) 
#temp is abstraction i.e. relation between accuracy and creativty more temo, more creativity



#create QA chain
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


#flask routes
@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Handle chat logic here
    return render_template('chat.html', bot_response="Hi! I am a medical chatbot. How can I assist you today?")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if file and file.filename.endswith(".pdf"):
        file.save("Medical_book.pdf")  # Overwrites the existing file
        return redirect(url_for("index"))
    return "Invalid file", 400


def format_response(response_text):
    lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
    return "\n".join(f"\u2022 {line}" for line in lines) if len(lines) > 1 else lines[0]


@app.route("/get", methods=["POST"])
def chatbot():
    msg = request.form.get("msg")
    if not msg:
        return "No message received.", 400
    print(f"User query: {msg}")
    try:
        # Vector similarity
        retrieved_docs = docsearch.similarity_search(msg, k=2)
        if not retrieved_docs:
            return "Sorry, I don’t have information on that in the provided document."
        # Model response
        result = qa.invoke({"query": msg})
        response_text = result.get("result", "No answer found.")
        formatted_response = format_response(response_text)
        # Saving to SQLite
        db_path = os.path.abspath("chat_history.db")
        print(f"Attempting to save chat to: {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("INSERT INTO chat (user, bot) VALUES (?, ?)", (msg, response_text))
            conn.commit()
            conn.close()
            print("✅ Chat saved to database.")
        except Exception as db_err:
            print(f"❌ DB Error: {db_err}")
        return formatted_response
    except Exception as e:
        import traceback
        print("❌ Chatbot Error:")
        traceback.print_exc()
        return f"An error occurred while processing your request: {e}"


@app.route('/view_history')
def view_history():
    try:
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute("SELECT timestamp, user, bot FROM chat ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()

        history = []
        for timestamp, user, bot in rows:
            try:
                time_str = format_timestamp(timestamp) if timestamp else "Unknown time"
            except Exception:
                time_str = "Unknown time"
            history.append({
                "user": user,
                "bot": bot,
                "time": time_str
            })
        return render_template("history.html", chats=history)
    except Exception as e:
        return f"Error loading chat history: {e}", 500


@app.route("/save_history", methods=["POST"])
def save_history():
    msg = request.form.get("msg")
    if not msg:
        return "No message received.", 400
    try:
        result = qa.invoke({"query": msg})
        response_text = result.get("result", "No answer found.")
        
        # Save to SQLite database
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute("INSERT INTO chat (user, bot) VALUES (?, ?)", (msg, response_text))
        conn.commit()
        conn.close()

        return format_response(response_text)
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"


@app.route("/delete_history", methods=["POST"])
def delete_history():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("DELETE FROM chat")
    conn.commit()
    conn.close()
    return redirect(url_for("view_history"))


def init_db():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user TEXT,
            bot TEXT
        )
    ''')
    conn.commit()
    conn.close()


if __name__ == '__main__':
     # Ensure the 'templates' directory exists for Flask to find chat.html
    os.makedirs('templates', exist_ok=True)
    # Ensure the 'model' directory exists for your Llama model
    os.makedirs('model', exist_ok=True) 
    init_db() # ← Call this to ensure DB table exists
    app.run(host="127.0.0.1", port=5050)
