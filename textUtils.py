import docx
from pypdf import PdfReader

def extract_text(file):
    text = ""

    if file.type == "application/pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text+" "

    elif file.type == "text/plain":
        text = file.read().decode("utf-8")

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text

    
