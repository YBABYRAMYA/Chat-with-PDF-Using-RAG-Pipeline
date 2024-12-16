# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
# from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic
# from langchain.agents import AgentExecutor, create_tool_calling_agent

# load_dotenv()
# embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
# def pdf_read(pdf_doc):
#     text = ""
#     for pdf in pdf_doc:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text



# def get_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def vector_store(text_chunks):
    
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_db")


# def get_conversational_chain(tools,ques):
#     #os.environ["ANTHROPIC_API_KEY"]=os.getenv["ANTHROPIC_API_KEY"]
#     llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"),verbose=True)

#     prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer""",
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )
#     tool=[tools]
#     agent = create_tool_calling_agent(llm, tool, prompt)

#     agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
#     response=agent_executor.invoke({"input": ques})
#     print(response)
#     st.write("Reply: ", response['output'])



# def user_input(user_question):
    
    
    
#     new_db = FAISS.load_local("faiss_db", embeddings,allow_dangerous_deserialization=True)
    
#     retriever=new_db.as_retriever()
#     retrieval_chain= create_retriever_tool(retriever,"pdf_extractor","This tool is to give answer to queries from the pdf")
#     get_conversational_chain(retrieval_chain,user_question)





# def main():
#     st.set_page_config("Chat PDF")
#     st.header("RAG based Chat with PDF")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = pdf_read(pdf_doc)
#                 text_chunks = get_chunks(raw_text)
#                 vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()










# from langchain.vectorstores import FAISS
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langchain.tools import Tool


# # Step 1: Initialize Embeddings
# # Using the HuggingFace model "sentence-transformers/all-MiniLM-L6-v2"
# def initialize_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# # Step 2: Load Documents
# # Example documents for demonstration purposes
# def load_documents():
#     return [
#         Document(page_content="The capital of France is Paris."),
#         Document(page_content="The Eiffel Tower is located in Paris, France."),
#         Document(page_content="Python is a popular programming language."),
#         Document(page_content="LangChain helps build language model applications."),
#     ]


# # Step 3: Create a Vector Store
# def create_vector_store(documents, embeddings):
#     return FAISS.from_documents(documents, embeddings)


# # Step 4: Create a Retriever Tool
# def create_retriever_tool(vectorstore):
#     retriever = vectorstore.as_retriever()
#     return Tool(
#         name="document_retriever",
#         func=retriever.get_relevant_documents,
#         description="Use this tool to retrieve relevant documents based on a query."
#     )


# # Step 5: Main Functionality
# if __name__ == "__main__":
#     # Initialize embeddings
#     embeddings = initialize_embeddings()

#     # Load documents
#     documents = load_documents()

#     # Create vector store
#     vectorstore = create_vector_store(documents, embeddings)

#     # Create retriever tool
#     retriever_tool = create_retriever_tool(vectorstore)

#     # User query input
#     query = input("Enter your query: ")

#     # Retrieve relevant documents
#     results = retriever_tool.func(query)

#     # Display results
#     print("\nRelevant Documents:")
#     if results:
#         for idx, result in enumerate(results, start=1):
#             print(f"{idx}. {result.page_content}")
#     else:
#         print("No relevant documents found.")





# from langchain.vectorstores import FAISS
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langchain.tools import Tool
# from langchain.text_splitter import CharacterTextSplitter
# import pdfplumber

# # Step 1: Extract Text from PDF
# def extract_text_from_pdf(pdf_path):
#     """Extracts text from a PDF file."""
#     extracted_text = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             extracted_text.append(page.extract_text())
#     return "\n".join(extracted_text)

# # Step 2: Split Extracted Text into Chunks
# def split_text_into_chunks(text, chunk_size=500, overlap=50):
#     """Splits the text into smaller chunks for vector storage."""
#     splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=overlap)
#     return splitter.split_text(text)

# # Step 3: Create Documents
# def create_documents(chunks):
#     """Converts text chunks into LangChain Document objects."""
#     return [Document(page_content=chunk) for chunk in chunks]

# # Step 4: Initialize Embeddings
# def initialize_embeddings():
#     """Initializes the HuggingFace embeddings model."""
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Step 5: Create Vector Store
# def create_vector_store(documents, embeddings):
#     """Creates a FAISS vector store from documents."""
#     return FAISS.from_documents(documents, embeddings)

# # Step 6: Create a Retriever Tool
# def create_retriever_tool(vectorstore):
#     """Wraps the retriever as a tool."""
#     retriever = vectorstore.as_retriever()
#     return Tool(
#         name="document_retriever",
#         func=retriever.get_relevant_documents,
#         description="Use this tool to retrieve relevant information from the PDF based on a query."
#     )

# # Step 7: Main Functionality
# if __name__ == "__main__":
#     # Path to the uploaded PDF file
#     # pdf_path = "C:\\Users\\RAMYA\\Downloads\\charts.pdf"
#     pdf_path = "C:\\Users\\RAMYA\\Downloads\\Y_Ramya_Resume_.pdf"

#     # Extract text from the PDF
#     pdf_text = extract_text_from_pdf(pdf_path)

#     # Split the text into smaller chunks
#     text_chunks = split_text_into_chunks(pdf_text)

#     # Create Document objects from the chunks
#     documents = create_documents(text_chunks)

#     # Initialize embeddings
#     embeddings = initialize_embeddings()

#     # Create vector store
#     vectorstore = create_vector_store(documents, embeddings)

#     # Create retriever tool
#     retriever_tool = create_retriever_tool(vectorstore)

#     # User query input
#     query = input("Enter your query: ")

#     # Retrieve relevant documents
#     results = retriever_tool.func(query)

#     # Display results
#     print("\nRelevant Information:")
#     if results:
#         for idx, result in enumerate(results, start=1):
#             print(f"{idx}. {result.page_content}\n")
#     else:
#         print("No relevant information found.")











from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.tools import Tool
from langchain.text_splitter import CharacterTextSplitter
import pdfplumber

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text.append(page.extract_text())
    return "\n".join(extracted_text)

# Step 2: Split Extracted Text into Chunks
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """Splits the text into smaller chunks for vector storage."""
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# Step 3: Create Documents
def create_documents(chunks):
    """Converts text chunks into LangChain Document objects."""
    return [Document(page_content=chunk) for chunk in chunks]

# Step 4: Initialize Embeddings
def initialize_embeddings():
    """Initializes the HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create Vector Store
def create_vector_store(documents, embeddings):
    """Creates a FAISS vector store from documents."""
    return FAISS.from_documents(documents, embeddings)

# Step 6: Create a Retriever Tool
def create_retriever_tool(vectorstore):
    """Wraps the retriever as a tool."""
    retriever = vectorstore.as_retriever()
    return Tool(
        name="document_retriever",
        func=retriever.get_relevant_documents,
        description="Use this tool to retrieve relevant information from the PDF based on a query."
    )

# Step 7: Main Functionality
if __name__ == "__main__":
    # User-provided PDF file path
    pdf_path = r"C:\\Users\\RAMYA\\Downloads\\charts.pdf"  # Update the path

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Split the text into smaller chunks
    text_chunks = split_text_into_chunks(pdf_text)

    # Create Document objects from the chunks
    documents = create_documents(text_chunks)

    # Initialize embeddings
    embeddings = initialize_embeddings()

    # Create vector store
    vectorstore = create_vector_store(documents, embeddings)

    # Create retriever tool
    retriever_tool = create_retriever_tool(vectorstore)

    # User query input
    query = input("Enter your query: ")

    # Retrieve relevant documents
    results = retriever_tool.func(query)

    # Display results
    print("\nRelevant Information:")
    if results:
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result.page_content}\n")
    else:
        print("No relevant information found.")
