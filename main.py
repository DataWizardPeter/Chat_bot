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

# Retrieve OpenAI API key from Streamlit Secrets
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please configure it in the Streamlit Cloud Secrets.")

# Initialize OpenAI LLM
llm = OpenAI(api_key=api_key, temperature=0.9, max_tokens=500)

# Function to encode image file to base64
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error encoding image to base64: {e}")
        return None

# Specify the local image path
image_path = "3d-rendering-financial-neon-bull (2).jpg"  # Replace with your image file name

# Generate the base64 string
base64_image = get_base64_of_bin_file(image_path)

# If base64 encoding fails, use a direct image URL (fallback)
if not base64_image:
    image_url = "https://example.com/your_image.jpg"  # Replace with your image URL
else:
    image_url = f"data:image/jpg;base64,{base64_image}"

# Add custom styles for background image
st.markdown(f"""
    <style>
    body {{
        background-image: url("{image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stButton > button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        font-size: 16px;
        padding: 10px;
    }}
    .stHeader {{
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
    }}
    .stSubheader {{
        color: #ff8c00;
        font-size: 24px;
        font-weight: bold;
    }}
    .stError {{
        color: #d32f2f;
        font-size: 18px;
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
                if "insufficient_quota" in str(e):
                    st.error("The API key used in this project has reached the free $5 quota limit. Please wait until the deployer upgrades the plan to continue.")
                else:
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
                if "insufficient_quota" in str(e):
                    st.error("The API key used in this project has reached the free $5 quota limit. Please wait until the deployer upgrades the plan to continue.")
                else:
                    st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("No FAISS index found. Please process the URLs first.")
