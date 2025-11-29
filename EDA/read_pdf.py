import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_path = "tweeter-ner-nlp.pdf"
    text = extract_text_from_pdf(pdf_path)
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Text extracted to pdf_content.txt")
