from typing import List
import re
def chunk_text(text:str,chunk_size:int=500,overlap:int=500)->List[str]:
    words=re.findall(r"\S+",text)
    if not words:
        return []
    
    chunks=[]
    i=0
    n=len(words)
    while i<n:
        j=min(i+chunk_size,n)
        chunk=" ".join(words[i:j])
        chunks.append(chunk)
        i=j-overlap
        if i<=0:
            i=j
    return chunks     
  
        
        