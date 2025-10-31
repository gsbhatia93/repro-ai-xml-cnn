'''
Eurlex Experiments 
'''


from xml_cnn_model import train_xmlcnn_from_dataframe, evaluate_xmlcnn


# eurlex_files_preprocess.py
# Reads train_labels.txt, train_texts.txt, test_labels.txt, test_texts.txt
# Outputs df_train / df_test in {"text": str, "labels": List[str]} format
# Also builds ltoi/itol and pos_weight for BCE, and (optionally) a validation split.

import os
import gzip
import json
import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

def _open_text(path: str):
    """Open .txt or .gz seamlessly, yield stripped lines (str)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line.strip()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line.strip()

def _read_split(labels_path: str, texts_path: str) -> pd.DataFrame:
    """Read aligned labels/texts files into a DataFrame."""
    labels_iter = _open_text(labels_path)
    texts_iter  = _open_text(texts_path)

    texts, labels = [], []
    for lbl_line, txt_line in zip(labels_iter, texts_iter):
        # labels are space-separated; keep underscores inside label names
        labs = [tok for tok in lbl_line.split() if tok]
        # text is already whitespace-tokenized; keep as a single string
        txt = " ".join(tok for tok in txt_line.split() if tok)
        texts.append(txt)
        labels.append(labs)

    # Safety: ensure same length; if file lengths differ, align to min length
    n_labels = len(labels)
    n_texts  = len(texts)
    n = min(n_labels, n_texts)
    if n_labels != n_texts:
        print(f"[warn] Mismatched lines: labels={n_labels}, texts={n_texts}. Truncating to {n}.")
        texts  = texts[:n]
        labels = labels[:n]

    return pd.DataFrame({"text": texts, "labels": labels})

def build_label_map(label_lists: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter
    c = Counter()
    for labs in label_lists:
        for y in labs:
            c[y] += 1
    # keep labels meeting min_freq
    items = [(y, f) for y, f in c.items() if f >= min_freq]
    # sort by frequency desc, then label name for stability
    items.sort(key=lambda x: (-x[1], x[0]))
    return {y: i for i, (y, _) in enumerate(items)}

def remap_and_filter_labels(df: pd.DataFrame, ltoi: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["labels"] = df["labels"].map(lambda labs: [y for y in labs if y in ltoi])
    # drop rows that lost all labels
    df = df[df["labels"].map(len) > 0].reset_index(drop=True)
    return df

def compute_pos_weight(df: pd.DataFrame, ltoi: Dict[str, int], clip: Tuple[float, float]=(0.5, 50.0)) -> np.ndarray:
    """pos_weight = (N - p) / p per label."""
    N = len(df)
    p = np.zeros(len(ltoi), dtype=np.float64)
    for labs in df["labels"]:
        for y in labs:
            p[ltoi[y]] += 1
    p = np.maximum(p, 1.0)
    pw = (N - p) / p
    lo, hi = clip
    return np.clip(pw, lo, hi)

def multilabel_iterative_split(idx: np.ndarray, y_lists: List[List[int]], test_size: float, seed: int=1337):
    """Simple greedy iterative stratification: returns (train_idx, test_idx)."""
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    n = len(idx)
    target = int(round(test_size * n))

    lab2s = defaultdict(set)
    for i, labs in enumerate(y_lists):
        for y in labs:
            lab2s[y].add(i)

    test = set()
    while len(test) < target:
        remaining = [i for i in range(n) if i not in test]
        if not remaining:
            break
        rare, cnt = None, 10**12
        for y, s in lab2s.items():
            c = sum(1 for i in s if i in remaining)
            if 0 < c < cnt:
                rare, cnt = y, c
        if rare is None:
            rng.shuffle(remaining)
            for i in remaining[:(target - len(test))]:
                test.add(i)
            break
        cand = [i for i in lab2s[rare] if i in remaining]
        rng.shuffle(cand)
        take = max(1, min(len(cand), target - len(test)))
        for i in cand[:take]:
            test.add(i)

    test_local = np.array(sorted(test), dtype=int)
    train_local = np.array([i for i in range(n) if i not in test], dtype=int)
    return idx[train_local], idx[test_local]

def add_valid_split(df_train: pd.DataFrame, ltoi: Dict[str, int], valid_size: float=0.1, seed: int=1337):
    idx = np.arange(len(df_train))
    y_lists = [[ltoi[y] for y in labs] for labs in df_train["labels"]]
    tr_idx, va_idx = multilabel_iterative_split(idx, y_lists, test_size=valid_size, seed=seed)
    return df_train.iloc[tr_idx].reset_index(drop=True), df_train.iloc[va_idx].reset_index(drop=True)

def load_eurlex_from_files(
    root_dir: str,
    train_labels: str = "train_labels.txt",
    train_texts: str  = "train_texts.txt",
    test_labels: str  = "test_labels.txt",
    test_texts: str   = "test_texts.txt",
    min_label_freq: int = 1,
    make_valid: bool = True,
    valid_size: float = 0.1,
    seed: int = 1337,
):
    """Main entry: returns df_train, df_valid (optional), df_test, ltoi, itol, pos_weight, stats."""
    train_labels_path = os.path.join(root_dir, train_labels)
    train_texts_path  = os.path.join(root_dir, train_texts)
    test_labels_path  = os.path.join(root_dir, test_labels)
    test_texts_path   = os.path.join(root_dir, test_texts)

    df_train = _read_split(train_labels_path, train_texts_path)
    df_test  = _read_split(test_labels_path,  test_texts_path)

    # Build label map on TRAIN ONLY (standard practice)
    ltoi = build_label_map(df_train["labels"].tolist(), min_freq=min_label_freq)
    itol = {v: k for k, v in ltoi.items()}

    # Remap + filter both splits; drop unlabeled rows
    df_train = remap_and_filter_labels(df_train, ltoi)
    df_test  = remap_and_filter_labels(df_test,  ltoi)

    # Optional validation split from train, with stratification
    if make_valid:
        df_train, df_valid = add_valid_split(df_train, ltoi, valid_size=valid_size, seed=seed)
    else:
        df_valid = pd.DataFrame(columns=["text","labels"])

    # pos_weight from TRAIN (after any filtering)
    pos_weight = compute_pos_weight(df_train, ltoi)

    stats = {
        "n_labels": len(ltoi),
        "train_docs": len(df_train),
        "valid_docs": len(df_valid),
        "test_docs": len(df_test),
        "avg_labels_per_train_doc": float(np.mean([len(x) for x in df_train["labels"]])),
        "min_label_freq": min_label_freq,
    }

    return df_train, df_valid, df_test, ltoi, itol, pos_weight, stats

# -------------- Example usage --------------
root = "data/eurlex"  # or local directory
df_train, df_valid, df_test, ltoi, itol, pos_weight, stats = load_eurlex_from_files(root, min_label_freq=1, make_valid=False)
print(stats)
print(df_train.head())

model, stoi, ltoi = train_xmlcnn_from_dataframe(
    df_train,
    data_name='eurlex',
    max_len=256,
    emb_dim=300,
    num_filters_per_size=128,
    p_chunks=6,
    bottleneck_dim=256,
    batch_size=16,
    max_epochs=80,    
)



# ===== A) Eval with in-memory model =====
# Assuming you still have: model, stoi, ltoi, and df_test
metrics = evaluate_xmlcnn(model, df_test, stoi, ltoi, batch_size=64)
print(metrics)
