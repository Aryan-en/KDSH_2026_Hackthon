import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import gc
import torch
# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# ML Libraries
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import faiss

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Utils
from tqdm import tqdm

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
# Allow forcing CUDA handling via env var `FORCE_CUDA=true` (for testing only).
# This does not create CUDA hardware; it only lets scripts behave as if CUDA
# availability was requested so you can test environment-specific code paths.
force_cuda = os.environ.get("FORCE_CUDA", "false").lower() in ("1", "true", "yes")
cuda_available = torch.cuda.is_available() or force_cuda
print(f"CUDA available: {cuda_available}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif force_cuda and not torch.cuda.is_available():
    print("FORCE_CUDA is set but PyTorch reports no CUDA devices; operations may fail.")

# Choose runtime device (will use CUDA only if PyTorch actually reports it).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up paths relative to workspace root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT
BOOKS_DIR = PROJECT_ROOT / 'Books-20260106T171745Z-1-001' / 'Books'

print("="*80)
print("LOADING DATA")
print("="*80)

# Load datasets
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

print(f"\nTrain: {len(train)} examples")
print(f"Test: {len(test)} examples")

# Analyze training data
print("\n--- Training Data Analysis ---")
print(f"Label distribution:\n{train['label'].value_counts()}")
print(f"\nBooks: {train['book_name'].unique()}")
print(f"Unique characters: {train['char'].nunique()}")

# Create validation split (80/20 stratified)
print("\n--- Creating Validation Split ---")
train_data, val_data = train_test_split(
    train, 
    test_size=0.2, 
    stratify=train['label'], 
    random_state=42
)

print(f"Train: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples")

# Display sample
print("\n--- Sample Training Example ---")
print(train_data.iloc[0])

print("\n Data loading and initial analysis complete. \n")

print("="*80)
print("NOVEL PROCESSING & CHUNKING")
print("="*80)

# Load novels
print("\n--- Loading Novels ---")
books = {}
book_paths = {
    'The Count of Monte Cristo': BOOKS_DIR / 'The Count of Monte Cristo.txt',
    'In Search of the Castaways': BOOKS_DIR / 'In search of the castaways.txt'
}

for book_name, path in book_paths.items():
    with open(path, 'r', encoding='utf-8') as f:
        books[book_name] = f.read()
    print(f"{book_name}: {len(books[book_name]):,} characters")

# Chunking function
def semantic_chunk(text, chunk_size=1000, overlap=150):
    """Chunk text with overlap, preserving paragraph boundaries"""
    from transformers import AutoTokenizer
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    
    # Split into paragraphs
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
        
        if current_length + para_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_paras = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_paras + [para]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_length = sum(len(tokenizer.encode(p, add_special_tokens=False)) for p in current_chunk)
        else:
            current_chunk.append(para)
            current_length += para_tokens
    
    # Add last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

# Chunk both novels
print("\n--- Chunking Novels ---")
book_chunks = {}
for book_name, text in books.items():
    print(f"Chunking {book_name}...")
    book_chunks[book_name] = semantic_chunk(text, chunk_size=1000, overlap=150)
    print(f"Created {len(book_chunks[book_name])} chunks")

print(f"\nTotal chunks: {sum(len(chunks) for chunks in book_chunks.values())}")


print("="*80)
print("LOADING LLM (PHI-2)")
print("="*80)

# Load Phi-2 (smaller, faster alternative to Mistral-7B)
print("\n--- Loading Phi-2 (2.7B, quantized) ---")
model_id = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "10GB", "cpu": "32GB"}
    )
    print("Phi-2 loaded successfully")
except Exception as e:
    print(f"Warning: Could not load Phi-2 with quantization: {e}")
    print("Loading without quantization (will use more VRAM)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("Phi-2 loaded on CPU")

try:
    print(f"  Model device: {model.device}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except:
    print("  (Memory info unavailable)")

def check_consistency_llm(claim, evidence_list):
    """Use LLM to check consistency with Chain-of-Thought"""
    
    # Format evidence (top 5 only)
    evidence_text = "\n\n".join([
        f"[{i+1}] {ev['text'][:500]}..." 
        for i, ev in enumerate(evidence_list[:5])
    ])
    
    # Chain-of-Thought prompt
    prompt = f"""Claim: {claim}

Relevant excerpts from the novel:
{evidence_text}

Think step-by-step:
1. What does the claim assert?
2. What do the excerpts say?
3. Are they compatible or contradictory?

Reasoning: [Your brief analysis]
Answer: [SUPPORTED or CONTRADICTED or NOT_MENTIONED]"""

    # Format for Mistral
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Parse verdict
    response_lower = response.lower()
    if 'contradicted' in response_lower or 'contradict' in response_lower:
        verdict = 'CONTRADICTED'
    elif 'supported' in response_lower or 'support' in response_lower:
        verdict = 'SUPPORTED'
    else:
        verdict = 'NOT_MENTIONED'
    
    return verdict, response

print("\nLLM reasoning engine ready")

# Test LLM
print("\n--- Testing LLM ---")
# Prepare a sample row and evidence for a smoke-test run
if 'val_data' in globals() and len(val_data) > 0:
    sample_row = val_data.iloc[0]
else:
    sample_row = train_data.iloc[0]

sample = sample_row.to_dict()

# Prepare evidence from the chunked novels; fall back safely if book missing
book_key = sample.get('book_name')
if book_key not in book_chunks:
    book_key = next(iter(book_chunks))

sample_evidence = [{'text': book_chunks[book_key][0]}]

sample_verdict, sample_response = check_consistency_llm(
    sample.get('content', ''),
    sample_evidence
)

print(f"LLM verdict: {sample_verdict}")
print(f"  Ground truth: {sample.get('label')}")
print(f"  Response preview: {sample_response[:200]}...")