import fitz  # PyMuPDF to read PDFs
import re
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- Configuration ---
CHINESE_REGEX = r'[\u4e00-\u9fff]'

# Setup robust paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) 
DATA_DIR = os.path.join(root_dir, "data") 

@dataclass
class PolicyChunk:
    text: str
    source: str
    page: int
    header_level: int
    heading_path: List[str]
    clause_number: Optional[str] = None
    metadata: Dict = None

class PDFIngestor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.filename = os.path.basename(pdf_path)
        
    def get_common_font_size(self):
        sizes = []
        for page in self.doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            sizes.append(round(s["size"], 1))
        return max(set(sizes), key=sizes.count) if sizes else 11.0

    def _clean_text(self, text: str) -> str:
        text = re.sub(CHINESE_REGEX, '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def determine_header_level(self, span: dict, body_size: float, text: str) -> int:
        text_letter_size = round(span['size'], 1)
        font = span['font'].lower()
        flags = span['flags']
        clean_text = text.strip()
        max_body_size = body_size + 10.0 

        is_flag_bold = bool(flags & 16)
        is_name_bold = any(x in font for x in ["bold", "black", "heavy", "med", "demi"])
        is_bold = is_flag_bold or is_name_bold

        if re.match(r'^[\u2022\u2023\u25E6\u2043\u2219\-\*]\s+', clean_text): 
            return 0
        if re.match(r'^\([a-zivx]+\)\s+', clean_text): 
            return 0

        if re.match(r'^SECTION\s+[\d\.]+\s*-', clean_text, re.IGNORECASE):
            if text_letter_size > body_size: return 1 
            else: return 3 
        elif clean_text.lower().endswith("section"):
            if text_letter_size >= max_body_size: return 1 
            elif text_letter_size >= body_size: return 2 
            else: return 0
        elif text_letter_size > (body_size + 0.1):
            if text_letter_size >= max_body_size: return 1 
            elif clean_text[0].isdigit(): return 4
            elif is_bold: return 2 
            else: return 3
        elif is_bold:
            return 4
        
        return 0

    def process(self) -> List[Dict]:
        chunks = []
        body_size = self.get_common_font_size()
        print(f"[{self.filename}] Body Size: {body_size}pt")
        
        hierarchy = {1: None, 2: None, 3: None, 4: None}
        current_buffer = [] 
        last_page_seen = 1
        last_header_level = 0

        for page_num, page in enumerate(self.doc, start=1):
            detected_page = page_num 
            footer_rect = fitz.Rect(0, page.rect.height - 50, page.rect.width, page.rect.height)
            footer_blocks = page.get_text("blocks", clip=footer_rect)
            
            for block in footer_blocks:
                text = block[4].strip()
                if text.isdigit():
                    detected_page = int(text)
                    break 

            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" not in b: continue
                if b["bbox"][1] < 30: continue
                if b["bbox"][1] > 800: continue

                for l in b["lines"]:
                    line_parts = []
                    max_size = 0
                    rep_span = None 

                    for s in l["spans"]:
                        if not s['text'].strip(): continue
                        line_parts.append(s['text'])
                        
                        if s['size'] > max_size:
                            max_size = s['size']
                            rep_span = s
                    
                    if not rep_span: continue 
                    
                    full_line_text = " ".join(line_parts)
                    clean_text = self._clean_text(full_line_text)
                    
                    if not clean_text: continue

                    level = self.determine_header_level(rep_span, body_size, clean_text)

                    if level > 0:
                        is_continuation = (
                            level == last_header_level and 
                            hierarchy[level] is not None and 
                            not clean_text[0].isdigit() and 
                            not current_buffer  
                        )

                        if is_continuation:
                            hierarchy[level] = hierarchy[level] + " " + clean_text
                        else:
                            if current_buffer:
                                self._save_chunk(chunks, current_buffer, last_page_seen, hierarchy)
                                current_buffer = []

                            hierarchy[level] = clean_text
                            last_header_level = level 
                            for i in range(level + 1, 5): hierarchy[i] = None
                    else:
                        current_buffer.append(clean_text)
                        last_page_seen = detected_page

        if current_buffer:
             self._save_chunk(chunks, current_buffer, last_page_seen, hierarchy)

        return chunks

    def _save_chunk(self, chunks_list, buffer, page, hierarchy):
        full_text = " ".join(buffer)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        if len(full_text) < 10: return 

        MAX_CHARS = 2400  
        OVERLAP = 400     
        
        current_heading_path = [v for k, v in sorted(hierarchy.items()) if v is not None]
        lowest_header = current_heading_path[-1] if current_heading_path else ""
        clause_match = re.search(r'(?:Section|Part)?\s*(\d+(\.\d+)*)', lowest_header, re.IGNORECASE)
        clause = clause_match.group(1) if clause_match else None
        
        active_levels = [k for k, v in hierarchy.items() if v is not None]
        current_level = max(active_levels) if active_levels else 0

        text_len = len(full_text)
        chunk_texts = []
        
        if text_len <= MAX_CHARS:
            chunk_texts.append(full_text)
        else:
            start = 0
            while start < text_len:
                end = start + MAX_CHARS
                if end < text_len:
                    last_space = full_text.rfind(' ', start, end)
                    if last_space != -1:
                        end = last_space
                segment = full_text[start:end]
                chunk_texts.append(segment)
                if end >= text_len: break
                start = end - OVERLAP
        
        for i, text_segment in enumerate(chunk_texts):
            chunk = PolicyChunk(
                text=text_segment,
                source=self.filename,
                page=page,
                header_level=current_level,
                heading_path=current_heading_path,
                clause_number=clause,
                metadata={
                    "combined_citation": f"{self.filename} > {'/'.join(current_heading_path)} (p.{page})"
                }
            )
            chunks_list.append(asdict(chunk))


# --- COMPLEX PIPELINE: For QBE Structured Policies ---
def process_folder_to_json(folder_name: str, output_filename: str):
    folder_path = os.path.join(DATA_DIR, folder_name)
    all_chunks = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: '{folder_path}' not found.")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ No PDFs found in {folder_name}/")
        return

    print(f"\n📂 Processing {len(pdf_files)} documents from '{folder_name}' using Hierarchical Parser...")

    for pdf in pdf_files:
        path = os.path.join(folder_path, pdf)
        try:
            ingestor = PDFIngestor(path)
            doc_chunks = ingestor.process()
            all_chunks.extend(doc_chunks)
            print(f"   ✅ {pdf}: Extracted {len(doc_chunks)} chunks.")
        except Exception as e:
            print(f"   ❌ Error {pdf}: {e}")

    output_path = os.path.join(root_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 Saved {len(all_chunks)} total chunks to '{output_filename}'.")


# --- SIMPLE PIPELINE: For General Educational Documents ---
def process_general_folder_to_json(folder_name: str, output_filename: str):
    folder_path = os.path.join(DATA_DIR, folder_name)
    all_chunks = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: '{folder_path}' not found.")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ No PDFs found in {folder_name}/")
        return

    print(f"\n📂 Processing {len(pdf_files)} documents from '{folder_name}' using Simple Text Splitter...")

    # Set up a clean, semantic text splitter for general knowledge
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for pdf in pdf_files:
        path = os.path.join(folder_path, pdf)
        try:
            doc = fitz.open(path)
            doc_chunks_count = 0
            
            for page_num, page in enumerate(doc, start=1):
                # Grab raw text, bypassing all the complex hierarchy logic
                raw_text = page.get_text("text").strip()
                raw_text = re.sub(CHINESE_REGEX, '', raw_text) # Keep your Chinese character cleaner
                
                if len(raw_text) < 50:
                    continue
                
                # Split the page text into semantic chunks
                split_texts = text_splitter.split_text(raw_text)
                
                for text_segment in split_texts:
                    # Package it using the exact same schema, but with "dummy" hierarchy data
                    chunk = PolicyChunk(
                        text=text_segment,
                        source=pdf,
                        page=page_num,
                        header_level=0,
                        heading_path=["General Industry Standard"], # Fake hierarchy so RAG doesn't crash
                        clause_number=None,
                        metadata={
                            "combined_citation": f"{pdf} > General Standard (p.{page_num})"
                        }
                    )
                    all_chunks.append(asdict(chunk))
                    doc_chunks_count += 1
                    
            print(f"   ✅ {pdf}: Extracted {doc_chunks_count} general chunks.")
        except Exception as e:
            print(f"   ❌ Error {pdf}: {e}")

    output_path = os.path.join(root_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 Saved {len(all_chunks)} total general chunks to '{output_filename}'.")


# --- Main Runner ---
if __name__ == "__main__":
    print("🚀 Starting PDF Extraction to JSON...")
    
    # 1. Process QBE documents into one JSON
    process_folder_to_json("qbe", "qbe_chunks.json")
    
    # 2. Process General documents into another JSON
    process_general_folder_to_json("general", "general_chunks.json")
    
    print("\n✅ Extraction complete!")