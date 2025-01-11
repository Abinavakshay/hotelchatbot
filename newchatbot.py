#1.
import pandas as pd

# List of dataset file names
datasets = ["bangalore.csv", "chennai.csv", "delhi.csv", "hyderabad.csv", "kolkata.csv", "mumbai.csv"]

# Initialize an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Loop through each file, add the city column, and concatenate
for file in datasets:
    df = pd.read_csv(file)
    city_name = file.split('.')[0]  # Extract the city name from the file name
    df['City'] = city_name  # Add city column
    combined_data = pd.concat([combined_data, df], ignore_index=True)

# Save combined data to a new CSV file
combined_data.to_csv("combined_hotel_data_with_city.csv", index=False)

# Check the result
print("Combined Data:")
print(combined_data.head())
#2.
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

# Load data and split for indexing
loader = CSVLoader(file_path="combined_hotel_data.csv")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Generate embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Example with OpenAI embeddings
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("hotel_vectorstore")
#3.
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp  # Or HuggingFacePipeline for Llama
from langchain.prompts import PromptTemplate

# Load the vectorstore
vectorstore = FAISS.load_local("hotel_vectorstore", embeddings)

# Initialize LLM
llm = LlamaCpp(model_path="path_to_llama_model.bin")  # Path to your Llama model

# Set up the retriever
retriever = vectorstore.as_retriever()

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="You are a helpful assistant. Answer the following hotel query using the given data: {query}"
)

# Create RetrievalQA chain
qa_chain = RetrievalQA(llm=llm, retriever=retriever, prompt_template=prompt_template)

# Query example
query = "Show me 5-star hotels in Chennai near the US Consulate."
response = qa_chain.run(query)
print(response)
#4.
import streamlit as st

st.title("Hotel Query Chatbot")
user_query = st.text_input("Ask me about hotels:")
if st.button("Submit"):
    response = qa_chain.run(user_query)
    st.write(response)
