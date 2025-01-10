from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

pinecone_key = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = pinecone_key

extracted_data=load_pdf_file(data='Data/')

for doc in extracted_data[:3]:  # Check the first 3 documents
    print(f"Document: {doc}")
    print(f"Page content: {doc.page_content[:200]}")  # Show the first 200 characters

text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=pinecone_key)

index_name = "medicalbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
)

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)