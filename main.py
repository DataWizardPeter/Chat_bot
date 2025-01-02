import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import base64

# Set Streamlit page config
st.set_page_config(page_title="BullBear BOT: Conquer the Bulls and Bears", page_icon="📊", layout="wide")

# Load environment variables
load_dotenv()

# Retrieve OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please configure it in your environment or secrets.")

# Initialize OpenAI LLM
llm = OpenAI(api_key=api_key, temperature=0.9, max_tokens=500)

# Function to encode an image file to Base64
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Dynamically generate the correct image path
image_path = os.path.join(os.path.dirname(__file__), "botimage2.jpg")

# Check if the image file exists and encode it
if os.path.exists(image_path):
    base64_image = get_base64_of_bin_file(image_path)
    if base64_image:
        image_url = f"data:image/jpg;base64,{base64_image}"
    else:
        st.error("Failed to encode the image to Base64.")
else:
    st.error(f"Image file not found: {image_path}")
    # Optional: Fallback to a public placeholder image
    image_url = "https://via.placeholder.com/1920x1080"

# Add custom styles for the background image
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

# Initialize Streamlit UI
st.title("BullBear BOT: Conquer the Bulls and Bears 📊")

# Sidebar setup
st.sidebar.title("News Article URLs")
st.sidebar.markdown("### Please input the URLs of the news articles you want to process.")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# Function to fetch and extract text from a URL using BeautifulSoup
def fetch_and_extract_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text
    except Exception as e:
        st.error(f"Error fetching content from {url}: {e}")
        return ""

# Processing URLs and extracting content
if process_url_clicked:
    if not any(urls):
        st.error("Please provide valid URLs for processing.")
    else:
        with st.spinner("Processing URLs and extracting content... please wait. ⏳"):
            try:
                all_documents = []
                for url in urls:
                    if url:
                        text = fetch_and_extract_text(url)
                        if text:
                            document = Document(page_content=text)
                            all_documents.append(document)

                if not all_documents:
                    st.error("No content extracted from the provided URLs.")
                else:
                    main_placeholder.text("Text Extraction Complete! Now Splitting Text... ✅✅✅")

                    # Split data
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=['\n\n', '\n', '.', ','],
                        chunk_size=1000
                    )
                    docs = text_splitter.split_documents(all_documents)

                    # Create embeddings and save to FAISS index
                    embeddings = OpenAIEmbeddings(api_key=api_key)
                    vectorstore_openai = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text("Embedding Vector Started Building... ✅✅✅")
                    time.sleep(2)

                    # Save the FAISS index to a pickle file
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore_openai, f)
            except requests.exceptions.RequestException as e:
                st.error(f"Error while fetching content from URLs: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

query = main_placeholder.text_input("Ask a Question about the News Articles:", placeholder="What do you want to know?")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            try:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                # Displaying results with styling
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(f"- {source}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("No FAISS index found. Please process the URLs first.")
