'''

RCV1 Experiment

'''

from xml_cnn_model import train_xmlcnn_from_dataframe, evaluate_xmlcnn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_rcv1
from scipy.sparse import csr_matrix

# ----------------------------------------------------------------------
# 1. Load the RCV1 dataset (vectorized form)
# ----------------------------------------------------------------------
rcv1 = fetch_rcv1(download_if_missing=True)
X: csr_matrix = rcv1.data
Y: csr_matrix = rcv1.target
sample_ids = rcv1.sample_id
print("Shapes:", X.shape, Y.shape)

# ----------------------------------------------------------------------
# 2. Build the official train/test split
# ----------------------------------------------------------------------
train_mask = sample_ids <= 26150 
test_mask  = sample_ids > 26150
print("Train docs:", train_mask.sum(), " Test docs:", test_mask.sum())

# ----------------------------------------------------------------------
# 3. Recover label (target) names safely
# ----------------------------------------------------------------------
# These are *always* available:
label_names = np.array(rcv1.target_names)
print("Label count:", len(label_names))

# ----------------------------------------------------------------------
# 4. Reconstruct vocabulary names
# ----------------------------------------------------------------------
# sklearn used a HashingVectorizer-like vocabulary in alphabetical order.
# We can rebuild it by reading the metadata file from the package.
# Fallback: synthesize dummy token names "term_0"... if unavailable.
try:
    from importlib.resources import files
    import io
    path = files("sklearn.datasets").joinpath("rcv1_vocab.txt")
    with io.TextIOWrapper(path.open("rb"), encoding="utf8") as f:
        vocab = np.array([line.strip() for line in f if line.strip()])
    print("Loaded vocab from rcv1_vocab.txt:", len(vocab), "terms")
except Exception:
    vocab = np.array([f"term_{i}" for i in range(X.shape[1])])
    print("Falling back to synthetic vocab of size", len(vocab))

# ----------------------------------------------------------------------
# 5. Turn sparse tf–idf rows → pseudo-text
# ----------------------------------------------------------------------
def row_to_pseudotext(row, vocab, max_tokens=400, scale=8.0):
    """Approximate a text string by repeating top-weighted tokens."""
    idx = row.indices
    vals = row.data
    if len(idx) == 0:
        return ""
    order = np.argsort(-vals)
    toks = []
    total = 0
    for j in order:
        token = vocab[idx[j]] if idx[j] < len(vocab) else f"term_{idx[j]}"
        reps = int(max(1, round(vals[j] * scale)))
        toks.extend([token] * reps)
        total += reps
        if total >= max_tokens:
            break
    return " ".join(toks)

def build_df(mask, limit=None):
    idxs = np.where(mask)[0]
    if limit is not None:
        idxs = idxs[:limit]
    texts, labels = [], []
    for i in idxs:
        texts.append(row_to_pseudotext(X.getrow(i), vocab))
        labs = Y.getrow(i).indices
        labels.append(label_names[labs].tolist())
    return pd.DataFrame({"text": texts, "labels": labels})

# ----------------------------------------------------------------------
# 6. Create subsets for manageable experiments
# ----------------------------------------------------------------------
df_train = build_df(train_mask, limit=50000)
df_test  = build_df(test_mask,  limit=881265) # todo: remove this limit 
print(f"Created {len(df_train)} train / {len(df_test)} test rows")
print(df_train.head())



model, stoi, ltoi = train_xmlcnn_from_dataframe(
    df_train,
    max_len=256,
    emb_dim=300,
    num_filters_per_size=128,
    p_chunks=6,
    bottleneck_dim=256,
    batch_size=16,
    max_epochs=6, 
    data_name='rcv1'
)

# # ===== A) Eval with in-memory model =====
# # Assuming you still have: model, stoi, ltoi, and df_test
metrics = evaluate_xmlcnn(model, df_test, stoi, ltoi, batch_size=64)
print(metrics)

