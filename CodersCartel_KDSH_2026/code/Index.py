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







# Runtime: approximately 10 minutes

print("="*80)
print("EMBEDDING & INDEXING")
print("="*80)

# Load embedding model
print("\n--- Loading Embedding Model ---")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
print("BGE-base-en-v1.5 loaded")

# Create FAISS index for each book
print("\n--- Creating FAISS Indices ---")
book_indices = {}
book_chunk_lists = {}

for book_name, chunks in book_chunks.items():
    print(f"\nEmbedding {book_name}...")
    
    # Embed chunks
    embeddings = embedder.encode(
        chunks, 
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    book_indices[book_name] = index
    book_chunk_lists[book_name] = chunks
    
    print(f"Indexed {len(chunks)} chunks ({dimension} dimensions)")

print("\nAll books indexed successfully!")






print("="*80)
print("RETRIEVAL SYSTEM")
print("="*80)

def retrieve_evidence(claim, book_name, char_name, top_k=10):
    """
    Retrieve relevant chunks using dual-query strategy
    
    Args:
        claim: Backstory claim text
        book_name: Name of the novel
        char_name: Character name
        top_k: Number of chunks to retrieve
    
    Returns:
        List of evidence dictionaries
    """
    # Get book index and chunks
    index = book_indices[book_name]
    chunks = book_chunk_lists[book_name]
    
    # Dual-query strategy
    query1 = claim
    query2 = f"{char_name}: {claim}"
    
    # Embed queries
    q1_emb = embedder.encode([query1])
    q2_emb = embedder.encode([query2])
    
    # Normalize
    faiss.normalize_L2(q1_emb)
    faiss.normalize_L2(q2_emb)
    
    # Search
    k = min(top_k, len(chunks))
    scores1, indices1 = index.search(q1_emb, k)
    scores2, indices2 = index.search(q2_emb, k)
    
    # Merge and deduplicate
    all_indices = list(indices1[0]) + list(indices2[0])
    all_scores = list(scores1[0]) + list(scores2[0])
    
    # Sort by score and deduplicate
    seen = set()
    evidence = []
    for idx, score in sorted(zip(all_indices, all_scores), key=lambda x: -x[1]):
        if idx not in seen:
            evidence.append({
                'text': chunks[idx],
                'score': float(score),
                'index': int(idx)
            })
            seen.add(idx)
        if len(evidence) >= top_k:
            break
    
    return evidence

print("Retrieval system ready")

# Test retrieval
print("\n--- Testing Retrieval ---")
sample = train_data.iloc[0]
sample_evidence = retrieve_evidence(
    sample['content'], 
    sample['book_name'], 
    sample['char'],
    top_k=5
)
print(f"Retrieved {len(sample_evidence)} chunks")
print(f"  Top score: {sample_evidence[0]['score']:.4f}")
print(f"  Preview: {sample_evidence[0]['text'][:200]}...")







print("="*80)
print("LOADING LLM (MISTRAL-7B)")
print("="*80)

# Load Mistral-7B with 4-bit quantization
print("\n--- Loading Mistral-7B-Instruct (4-bit) ---")
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    # When some modules are offloaded to CPU/disk, keep them in fp32 on CPU
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

try:
    # Try loading with 4-bit quantization and automatic device placement.
    # Provide max_memory so the auto device mapper can offload properly.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "20GB", "cpu": "64GB"}
    )
    print("Mistral-7B loaded successfully (quantized)")
except Exception as e:
    print(f"Warning: Could not load Mistral-7B with quantization: {e}")
    print("Falling back to CPU-only (no quantization). This will be slower and use more RAM.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("Mistral-7B loaded on CPU")

try:
    print(f"  Model device: {model.device}")
    if torch.cuda.is_available():
        print(f"  Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except Exception:
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

    # Robust token placement for sharded / device-mapped models
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device("cpu")

    # Debug prints to help diagnose hangs or slow CPU generation
    print("DEBUG: model primary device:", model_device)
    print("DEBUG: hf_device_map:", getattr(model, 'hf_device_map', None))
    print("DEBUG: model.device attribute:", getattr(model, 'device', None))

    inputs = tokenizer(formatted, return_tensors="pt")
    # Move tensors to the device where model parameters live (works with sharded models)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print("Generation error:", e)
            return 'NOT_MENTIONED', str(e)
    
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
# print("\n--- Testing LLM ---")
print("DEBUG before LLM test: primary model device check")
try:
    print("DEBUG model primary device:", next(model.parameters()).device)
except Exception:
    print("DEBUG model has no parameters or cannot access primary device")
print("DEBUG hf_device_map:", getattr(model, 'hf_device_map', None))
print("DEBUG model.device attribute:", getattr(model, 'device', None))

sample_verdict, sample_response = check_consistency_llm(
    sample['content'], 
    sample_evidence
)
print(f"LLM verdict: {sample_verdict}")
print(f"  Ground truth: {sample['label']}")
print(f"  Response preview: {sample_response[:200]}...")











print("="*80)
print("FEATURE EXTRACTION")
print("="*80)

def extract_features(row, evidence_list, llm_verdict):
    """
    Extract 12 features for classifier
    
    Features:
    - Retrieval: max/mean/min similarity, num_high_sim
    - LLM: verdict flags (3)
    - Character: total/max mentions
    - Claim: length
    - Book: is_monte_cristo
    - Evidence: num_evidence
    """
    claim = row['content']
    char_name = row['char']
    
    # Similarity scores
    similarities = [ev['score'] for ev in evidence_list]
    
    # Character mentions
    char_mentions = [
        ev['text'].lower().count(char_name.lower()) 
        for ev in evidence_list
    ]
    
    features = {
        # Retrieval features (4)
        'max_similarity': max(similarities) if similarities else 0,
        'mean_similarity': np.mean(similarities) if similarities else 0,
        'min_similarity': min(similarities) if similarities else 0,
        'num_high_sim': sum(1 for s in similarities if s > 0.7),
        
        # LLM features (3)
        'llm_verdict_supported': 1 if llm_verdict == 'SUPPORTED' else 0,
        'llm_verdict_contradicted': 1 if llm_verdict == 'CONTRADICTED' else 0,
        'llm_verdict_not_mentioned': 1 if llm_verdict == 'NOT_MENTIONED' else 0,
        
        # Character features (2)
        'total_char_mentions': sum(char_mentions),
        'max_char_mentions': max(char_mentions) if char_mentions else 0,
        
        # Claim features (1)
        'claim_length': len(claim.split()),
        
        # Book features (1)
        'is_monte_cristo': 1 if row['book_name'] == 'The Count of Monte Cristo' else 0,
        
        # Evidence quality (1)
        'num_evidence': len(evidence_list),
    }
    
    return features

print("Feature extraction function ready")
print("\n--- Feature List (12 total) ---")
sample_features = extract_features(sample, sample_evidence, sample_verdict)
for i, (feat, val) in enumerate(sample_features.items(), 1):
    print(f"{i:2d}. {feat:30s} = {val}")





print("="*80)
print("PROCESSING TRAINING DATA")
print("="*80)
print("\nEstimated time: 30-60 minutes")
print("Tip: This is a good time for a coffee break!\n")

train_features = []
train_labels = []

for idx, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Training"):
    try:
        # Retrieve evidence
        evidence = retrieve_evidence(
            row['content'], 
            row['book_name'], 
            row['char'],
            top_k=10
        )
        
        # Get LLM verdict
        llm_verdict, _ = check_consistency_llm(row['content'], evidence)
        
        # Extract features
        features = extract_features(row, evidence, llm_verdict)
        train_features.append(features)
        
        # Label (convert to binary)
        label = 1 if row['label'] == 'consistent' else 0
        train_labels.append(label)
        
        # Clear GPU cache periodically
        if idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        print(f"\nError processing row {idx}: {e}")
        # Use default features on error
        train_features.append({k: 0 for k in ['max_similarity', 'mean_similarity', 'min_similarity', 
                                                'num_high_sim', 'llm_verdict_supported', 
                                                'llm_verdict_contradicted', 'llm_verdict_not_mentioned',
                                                'total_char_mentions', 'max_char_mentions', 
                                                'claim_length', 'is_monte_cristo', 'num_evidence']})
        train_labels.append(0)

# Convert to DataFrame
train_features_df = pd.DataFrame(train_features)
train_labels = np.array(train_labels)

print(f"\nTraining features extracted: {train_features_df.shape}")
print(f"Labels: {len(train_labels)}")
print(f"\nLabel distribution: {pd.Series(train_labels).value_counts().to_dict()}")

# Save intermediate results
train_features_df.to_csv('train_features.csv', index=False)
np.save('train_labels.npy', train_labels)
print("\nSaved intermediate results (train_features.csv, train_labels.npy)")




print("="*80)
print("TRAINING CLASSIFIER")
print("="*80)

print("\n--- Training Random Forest ---")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

clf.fit(train_features_df, train_labels)
print("\nClassifier trained successfully")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': train_features_df.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 10 Most Important Features ---")
print(feature_importance.head(10).to_string(index=False))



print("="*80)
print("VALIDATION")
print("="*80)

print("\n--- Processing Validation Data ---")
val_features = []
val_labels = []

for idx, row in tqdm(val_data.iterrows(), total=len(val_data), desc="Validation"):
    try:
        evidence = retrieve_evidence(row['content'], row['book_name'], row['char'])
        llm_verdict, _ = check_consistency_llm(row['content'], evidence)
        features = extract_features(row, evidence, llm_verdict)
        val_features.append(features)
        val_labels.append(1 if row['label'] == 'consistent' else 0)
        
        if idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        print(f"\nError processing validation row {idx}: {e}")
        val_features.append({k: 0 for k in train_features_df.columns})
        val_labels.append(0)

val_features_df = pd.DataFrame(val_features)
val_labels = np.array(val_labels)

print(f"\nValidation features extracted: {val_features_df.shape}")

# Predict
print("\n--- Making Predictions ---")
val_preds = clf.predict(val_features_df)

# Evaluate
print("\n" + "="*80)
print("VALIDATION RESULTS")
print("="*80)
accuracy = accuracy_score(val_labels, val_preds)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if accuracy >= 0.85:
    print("EXCELLENT! Target accuracy achieved (>=85%)")
elif accuracy >= 0.75:
    print("GOOD! Above baseline (>=75%)")
else:
    print("Below target. Consider tuning parameters.")

print("\n--- Classification Report ---")
print(classification_report(val_labels, val_preds, target_names=['contradict', 'consistent']))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(val_labels, val_preds)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")





print("="*80)
print("TEST PREDICTIONS")
print("="*80)

print("\n--- Processing Test Data ---")
test_features = []

for idx, row in tqdm(test.iterrows(), total=len(test), desc="Test"):
    try:
        evidence = retrieve_evidence(row['content'], row['book_name'], row['char'])
        llm_verdict, _ = check_consistency_llm(row['content'], evidence)
        features = extract_features(row, evidence, llm_verdict)
        test_features.append(features)
        
        if idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        print(f"\nError processing test row {idx}: {e}")
        test_features.append({k: 0 for k in train_features_df.columns})

test_features_df = pd.DataFrame(test_features)

print(f"\nTest features extracted: {test_features_df.shape}")

# Predict
print("\n--- Making Final Predictions ---")
test_preds = clf.predict(test_features_df)

# Create submission
submission = pd.DataFrame({
    'story_id': test['id'],
    'prediction': test_preds
})

submission.to_csv('results.csv', index=False)

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nSubmission saved to: results.csv")
print(f"Total test predictions: {len(test_preds)}")
print(f"\nPrediction distribution:")
print(f"  Consistent (1): {(test_preds == 1).sum()}")
print(f"  Contradict (0): {(test_preds == 0).sum()}")

print("\n--- First 10 Predictions ---")
print(submission.head(10))