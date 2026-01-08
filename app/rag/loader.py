from typing import List
def load_text_file(path:str):
    with open(path,"r",encoding="utf8") as f:
        return f.read()
    
    
def load_pdf(path:str)->str:
    try:
        from PyPDF2 import pdfReader
    except Exception as e:
        raise RuntimeError("PyPdf is required") from e    
    reader=pdfReader(path)
    pages=[]
    for page in reader.pages:
        pages.append(page.extract_text() or "")
        
    return "\n".join(pages)


def load_files(paths:List[str])->List[str]:
    texts=[]
    for p in paths:
        if p.lower().endswith(".pdf"):
            texts.append(load_pdf(p))
        else:
            texts.append(load_text_file(p))
    return texts            
