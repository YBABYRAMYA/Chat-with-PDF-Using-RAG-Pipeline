---

# 📚 Chat with PDF 🤖

A Python-based application to interact with PDF documents using a Retrieval-Augmented Generation (RAG) pipeline. It scrapes, processes, and generates context-rich responses based on user queries.

## 🚀 Features

- 📰 **Crawl & Scrape**: Scrapes content from URLs and stores the data.
- 🔍 **Search**: Queries a vector store (FAISS) for relevant information.
- 🧠 **Contextual Responses**: Uses LLMs (e.g., GPT) to generate answers from the scraped content.
- 🌐 **Supports PDFs**: Extracts text and metadata from PDF files.

## 💻 Installation 💻

### Prerequisites

Make sure you have Python 3.7+ installed on your system. You can check your version using:

```bash
python --version
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Chat-with-PDF.git
cd Chat-with-PDF
```

### Step 2: Install Dependencies

You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Step 3: Install FAISS (Optional)

If you're using FAISS for vector search, install the appropriate version:

- For CPU:
  ```bash
  pip install faiss-cpu
  ```
- For GPU (if you're using CUDA):
  ```bash
  pip install faiss-gpu
  ```

### Step 4: Install `sentence-transformers`

This is required for generating embeddings:

```bash
pip install sentence-transformers
```

---

## ⚙️ Usage

### Step 1: Scrape Website

To scrape a website, use the `scrape_website(url)` function. Here’s an example:

```python
url = "https://example.com"
scraped_data = scrape_website(url)
print(scraped_data)
```

### Step 2: Process PDF

To process a PDF file, use the following:

```python
import pdfplumber

with pdfplumber.open("example.pdf") as pdf:
    for page in pdf.pages:
        print(page.extract_text())
```

### Step 3: Query the System

Once the data is processed and stored in the vector store, you can query the system like so:

```python
query = "What is the capital of France?"
response = query_system(query)
print(response)
```

---

## 🛠️ Technologies Used 🛠️

- **Python** 🐍
- **FAISS** for vector storage 🔐
- **Hugging Face** for embeddings 🤗
- **BeautifulSoup** for web scraping 🌐
- **pdfplumber** for PDF extraction 📄
- **LangChain** for the pipeline ⚡

---

## 🤖 How It Works

1. **Data Ingestion** 📥:
   - Scrape website content or process PDF files.
   - Segment and store content as embeddings in a vector store (FAISS).

2. **Query Handling** 🧐:
   - Convert the user's query into embeddings.
   - Perform a similarity search in the vector store.
   
3. **Response Generation** 💬:
   - Use a Language Model (e.g., GPT) to generate responses based on the retrieved data.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

- **Your Name** 🖋️
- **Email**: yalagandulababyramya@gmail.com 📧
- **GitHub**: [Github](https://github.com/YBABYRAMYA) 🐙

---

