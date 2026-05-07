import os
import streamlit as st
from docx import Document
from pypdf import PdfReader

from langchain.schema import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="MFU Grant Assistant using RAG",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# GROUP INFO (แก้ข้อมูลจริงก่อนส่ง)
# ==================================================
GROUP_NO = "BDA_Project2_YourGroupNo"

GROUP_MEMBERS = [
    {"student_id": "65XXXXXXXX", "name": "Your Name"},
    {"student_id": "65XXXXXXXX", "name": "Member 2"},
    {"student_id": "65XXXXXXXX", "name": "Member 3"},
]


DATASET_DIR = "dataset"


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.title("📘 Project Info")

    st.markdown(f"### Group No.\n**{GROUP_NO}**")

    st.markdown("### Members")
    for member in GROUP_MEMBERS:
        st.write(f"- {member['student_id']} : {member['name']}")

    st.divider()

    st.markdown("### AI Concept")
    st.write("""
    This project applies Retrieval-Augmented Generation (RAG)
    to answer questions from MFU textbook grant datasets,
    including:
    - Grant Type 1
    - Grant Type 2 (eBook)
    - Application steps
    - Evaluation
    - Copyright
    """)

    st.divider()

    st.markdown("### Example Questions")
    st.write("""
    - ทุนประเภท 1 กับ 2 ต่างกันอย่างไร  
    - ได้เงินสนับสนุนเมื่อไหร่  
    - ถ้าไม่ผ่านระดับดีต้องทำอย่างไร  
    - ลิขสิทธิ์กี่ปี  
    """)


# ==================================================
# FILE READERS
# ==================================================
def read_docx(file_path):
    try:
        doc = Document(file_path)
        text = []

        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text.strip())

        return "\n".join(text)

    except Exception as e:
        return f"Error reading DOCX: {e}"


def read_pdf(file_path):
    text = []

    try:
        reader = PdfReader(file_path)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

    except Exception as e:
        text.append(f"Error reading PDF: {e}")

    return "\n".join(text)


# ==================================================
# IMAGE SUMMARY (จาก Infographic)
# ==================================================
IMAGE_KNOWLEDGE = """
ทุนประเภทที่ 1:
- เพื่อขอตำแหน่งทางวิชาการ
- มหาวิทยาลัยสนับสนุน 30,000 บาท
- ผ่านสำนักวิชา
- ผู้ทรงคุณวุฒิ 2 คน
- ต้องผ่านระดับดีขึ้นไป
- ได้ค่าลิขสิทธิ์ 30%

ทุนประเภทที่ 2:
- eBook เพื่อการจำหน่าย
- ผู้เขียนส่งเอง
- ผู้เขียนจ่ายค่าผู้ทรงคุณวุฒิ
- ต้องผ่านระดับดีขึ้นไป
- MLii จัดทำ eBook
- ได้รายได้ 80% หลังหักค่าใช้จ่าย
- มอบลิขสิทธิ์ 5 ปี
"""


# ==================================================
# LOAD DOCUMENTS
# ==================================================
def load_documents():
    documents = []

    if not os.path.exists(DATASET_DIR):
        return documents

    for root, dirs, files in os.walk(DATASET_DIR):

        for file in files:

            path = os.path.join(root, file)

            content = ""

            if file.lower().endswith(".docx"):
                content = read_docx(path)

            elif file.lower().endswith(".pdf"):
                content = read_pdf(path)

            else:
                continue

            if content.strip():
                documents.append(
                    LangchainDocument(
                        page_content=content,
                        metadata={"source": file}
                    )
                )

    documents.append(
        LangchainDocument(
            page_content=IMAGE_KNOWLEDGE,
            metadata={"source": "ทุนประเภทที่1-2 Infographic Summary"}
        )
    )

    return documents


# ==================================================
# VECTORSTORE
# ==================================================
@st.cache_resource
def create_vectorstore():

    documents = load_documents()

    if len(documents) == 0:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# ==================================================
# ANSWER FUNCTION
# ==================================================
def get_answer(question, vectorstore):

    docs = vectorstore.similarity_search(question, k=4)

    if not docs:
        return "ไม่พบข้อมูลในเอกสาร", []

    answer = "### สรุปคำตอบจาก Dataset\n"

    sources = []

    for i, doc in enumerate(docs, start=1):

        answer += f"\n\nข้อมูลอ้างอิง {i}:\n"
        answer += doc.page_content[:800]

        source = doc.metadata.get("source", "Unknown")
        sources.append(source)

    return answer, sources


# ==================================================
# MAIN UI
# ==================================================
st.title("📚 MFU Grant Assistant using RAG")

st.write("""
ระบบ AI Chatbot สำหรับตอบคำถามเกี่ยวกับ:
- ทุนตำราเพื่อขอตำแหน่งทางวิชาการ
- ทุนตำรา eBook เพื่อการจำหน่าย
- ขั้นตอนการขอทุน
- การประเมิน
- ค่าลิขสิทธิ์
""")


vectorstore = create_vectorstore()

if vectorstore is None:
    st.error("ไม่พบ dataset กรุณาตรวจสอบโฟลเดอร์ dataset")
    st.stop()


question = st.text_input(
    "กรอกคำถามของคุณ:",
    placeholder="เช่น ทุนประเภทที่ 1 กับประเภทที่ 2 ต่างกันอย่างไร?"
)


if question:

    with st.spinner("กำลังวิเคราะห์ข้อมูลจาก Dataset..."):

        answer, sources = get_answer(question, vectorstore)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources Used")

    unique_sources = list(set(sources))

    for src in unique_sources:
        st.info(src)


st.divider()

st.caption("Developed using Vibecode + RAG + Streamlit")
