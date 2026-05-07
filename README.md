# MFU Grant Assistant using RAG

## Project Name
MFU Grant Assistant: AI Chatbot for Textbook Grant Support using RAG

## Group No.
BDA_Project2_yourGroupNo

## Project Description
This project develops an AI chatbot using Retrieval-Augmented Generation (RAG) to answer questions from the provided dataset about MFU textbook grants, eBook grants, application procedures, conditions, evaluation criteria, copyright, and benefits.

## Dataset
The dataset includes:
Q&A ทุนตำรา.docx
เอกสารและแบบฟอร์มการขอรับทุนตำรา.docx
ประกาศตำราเพื่อการจำหน่าย 68.pdf
ประกาศทุนตำราเพื่อขอตำแหน่ง 2566.pdf
ขั้นตอนทุนประเภทที่ 1.png
ขั้นตอนทุนประเภทที่ 2.png

## AI Concept
The system applies RAG by:
1. Loading documents from the dataset
2. Splitting text into chunks
3. Creating embeddings
4. Storing embeddings in FAISS vector database
5. Retrieving relevant chunks based on user questions
6. Displaying answers through Streamlit web application

## Tools
Python
Streamlit
LangChain
FAISS
Sentence Transformers
