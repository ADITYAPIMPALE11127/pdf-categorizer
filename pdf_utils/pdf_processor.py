import io
import PyPDF2
import zipfile

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extract text from PDF bytes
    Returns: Text string or empty string if failure
    """
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        full_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + " "
        
        return full_text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_all_pdfs_from_zip(zip_file):
    """
    Extract text from all PDFs in a ZIP file
    Returns: (pdf_texts, pdf_names)
    """
    pdf_texts = []
    pdf_names = []
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        pdf_files = [f for f in all_files if f.endswith('.pdf')]
        
        for pdf_name in pdf_files:
            try:
                pdf_bytes = zip_ref.read(pdf_name)
                text = extract_text_from_pdf_bytes(pdf_bytes)
                
                if text:
                    pdf_texts.append(text)
                    pdf_names.append(pdf_name)
                else:
                    print(f"No text extracted from: {pdf_name}")
                    
            except Exception as e:
                print(f"Error reading {pdf_name}: {e}")
    
    return pdf_texts, pdf_names

def get_pdf_stats(pdf_texts):
    """
    Get statistics about extracted PDFs
    """
    if not pdf_texts:
        return {}
    
    stats = {
        'total_docs': len(pdf_texts),
        'avg_chars': sum(len(t) for t in pdf_texts) // len(pdf_texts),
        'total_words': sum(len(t.split()) for t in pdf_texts),
        'doc_lengths': [len(t) for t in pdf_texts]
    }
    
    return stats