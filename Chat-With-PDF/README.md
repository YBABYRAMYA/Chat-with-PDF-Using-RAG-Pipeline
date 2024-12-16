# 📚 **Chat-with-PDF** 🤖  
### *Extract, Split, and Query PDF Files with Ease*

---

## 🛠️ **Overview** 🛠️

This Python project extracts text from a PDF file, splits it into smaller chunks, stores them in a FAISS vector store, and enables querying to retrieve relevant information using LangChain and HuggingFace embeddings.

---

## 🚀 **Key Features** 🚀

- 📄 **PDF Text Extraction**: Seamlessly extract text from PDFs using `pdfplumber`.
- 🔪 **Text Chunking**: Split the text into manageable chunks for processing.
- 🧠 **Embeddings**: Generate embeddings using HuggingFace’s `all-MiniLM-L6-v2` model.
- ⚡ **Vector Search**: Store and retrieve data quickly with **FAISS** for similarity search.
- 🔍 **Query Functionality**: Ask questions about the PDF and get relevant information.

---

## 📝 **File Structure** 📝

The project structure follows clean organization for readability and scalability:

```
Chat-with-PDF/
│
├── app.py              # Main Python script
├── .env                # Environment file for configurations
├── requirements.txt    # Project dependencies
├── LICENSE             # Project license file
├── README.md           # Project documentation
└── data/               # Folder for storing PDF files
    └── example.pdf     # Example PDF for demonstration
```

---

## 💻 **Setup & Installation** 💻

Follow the steps below to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/Chat-with-PDF.git
   cd Chat-with-PDF
   ```

2. **Install Dependencies**:
   Make sure Python 3.8+ is installed. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   **Dependencies**:
   - `langchain`
   - `langchain-community`
   - `pdfplumber`
   - `sentence-transformers`
   - `faiss-cpu`

3. **Add PDF File**:
   Place your PDF files in the `data/` directory.

---

## ⚙️ **How It Works**

### 1️⃣ **Extract Text from PDF**  1️⃣
The `extract_text_from_pdf()` function uses `pdfplumber` to extract text page-by-page.

```python
def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text.append(page.extract_text())
    return "\n".join(extracted_text)
```

### 2️⃣ **Split Text into Chunks** 2️⃣
The `CharacterTextSplitter` splits extracted text into smaller chunks for vector storage.

```python
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
```

### 3️⃣ **Generate Embeddings & Store in FAISS** 3️⃣
Text chunks are converted into embeddings using HuggingFace and stored in FAISS for efficient retrieval.

---

## 🖥️ **Run the Project** 🖥️

1. **Edit PDF Path**: Update the file path in `app.py`:
   ```python
   pdf_path = r"data/example.pdf"
   ```

2. **Run the Script**:
   ```bash
   python app.py
   ```

3. **Ask Questions**: Enter a query when prompted to retrieve relevant content from the PDF.

---

## 📊 **Example Output** 📊

```plaintext
Query: "What does the chart in the PDF represent?"
Response:
"18% Arts, entertainment, 59% recreation, accommodation, and food services."
```

---

## 🔧 **Dependencies** 🔧

Ensure these libraries are installed:

- `langchain`
- `pdfplumber`
- `sentence-transformers`
- `faiss-cpu`

You can install them all using:
```bash
pip install langchain langchain-community pdfplumber sentence-transformers faiss-cpu
```

---

## 🤝 **Contributing** 🤝

Contributions are welcome! Open issues or pull requests for improvements.

---

### **License**  
📄 Licensed under the MIT License. 📄

---

✨ **Enjoy querying your PDFs with AI-powered tools!** 🚀
