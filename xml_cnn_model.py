import kagglehub
import numpy as np
import torch


# Download latest version
path = kagglehub.dataset_download("thanakomsn/glove6b300dtxt")

print("Path to dataset files:", path)


def load_glove_embeddings(glove_path: str, stoi: dict, emb_dim: int = 300):
    """Return a tensor aligned to your stoi (vocab)."""
    embeddings = np.random.normal(scale=0.6, size=(len(stoi), emb_dim)).astype(np.float32)
    pad_id = stoi.get("<pad>", 0)
    embeddings[pad_id] = 0.0

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word, values = parts[0], parts[1:]
            if word in stoi and len(values) == emb_dim:
                embeddings[stoi[word]] = np.asarray(values, dtype=np.float32)
    return torch.tensor(embeddings)



# XML-CNN (SIGIR'17) — PyTorch reference implementation for XMTC
# Train directly from a pandas DataFrame with columns: text (str), labels (Iterable[str|int])

import math
import re
import random
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

############################################################
# 1) Tokenization & Vocab
############################################################

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u00C0-\u024F]+")

def simple_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [t.lower() for t in TOKEN_RE.findall(text)]

def build_vocab(texts: Iterable[str], min_freq: int = 2, max_vocab: int = 100_000,
                specials: List[str] = ["<pad>", "<unk>"]) -> Dict[str, int]:
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(simple_tokenize(t))
    # sort by freq then alpha for stability
    items = sorted([kv for kv in cnt.items() if kv[1] >= min_freq],
                   key=lambda x: (-x[1], x[0]))
    if max_vocab is not None:
        items = items[: max(0, max_vocab - len(specials))]
    stoi = {sp: i for i, sp in enumerate(specials)}
    for w, _ in items:
        stoi[w] = len(stoi)
    return stoi

def vectorize(tokens: List[str], stoi: Dict[str, int]) -> List[int]:
    unk_id = stoi.get("<unk>", 1)
    return [stoi.get(tok, unk_id) for tok in tokens]

############################################################
# 2) Label mapping
############################################################

def build_label_map(label_lists: Iterable[Iterable], max_labels: Optional[int] = None) -> Dict:
    from collections import Counter
    c = Counter()
    for labels in label_lists:
        if labels is None: 
            continue
        for y in labels:
            c[y] += 1
    items = sorted(c.items(), key=lambda x: (-x[1], str(x[0])))
    if max_labels is not None:
        items = items[:max_labels]
    return {lbl: i for i, (lbl, _) in enumerate(items)}

############################################################
# 3) Dataset & Collate
############################################################

class XMTCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stoi: Dict[str, int], ltoi: Dict,
                 text_col: str = "text", labels_col: str = "labels",
                 max_len: int = 512):
        self.texts = df[text_col].tolist()
        self.labels = df[labels_col].tolist()
        self.stoi = stoi
        self.ltoi = ltoi
        self.max_len = max_len
        self.pad_id = stoi.get("<pad>", 0)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = simple_tokenize(self.texts[idx])
        ids = vectorize(tokens, self.stoi)
        if self.max_len is not None:
            ids = ids[:self.max_len]
        # label multi-hot
        y = torch.zeros(len(self.ltoi), dtype=torch.float32)
        labs = self.labels[idx] or []
        for lbl in labs:
            if lbl in self.ltoi:
                y[self.ltoi[lbl]] = 1.0
            elif isinstance(lbl, (int, np.integer)) and lbl < len(self.ltoi):
                # if labels already numeric and aligned (optional)
                y[int(lbl)] = 1.0
        return torch.tensor(ids, dtype=torch.long), y

def pad_collate(batch, pad_id: int):
    seqs, ys = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :s.size(0)] = s
    y = torch.stack(ys, dim=0)
    return padded, y

############################################################
# 4) XML-CNN Model
############################################################

class DynamicMaxPool(nn.Module):
    """
    Split the temporal dimension into p chunks and take max within each chunk.
    Input: (B, C, T)  -> Output: (B, C, p)
    """
    def __init__(self, p: int):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        p = self.p
        # if T not divisible by p, pad with -inf so max ignores padded tail
        pad = (p - (T % p)) % p
        if pad:
            # pad at end along temporal dimension
            pad_tensor = x.new_full((B, C, pad), float("-inf"))
            x = torch.cat([x, pad_tensor], dim=2)
            T = T + pad
        # reshape to (B, C, p, T//p) and max over last dim -> (B, C, p)
        x = x.view(B, C, p, T // p)
        x, _ = torch.max(x, dim=3)
        return x

class XMLCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        emb_dim: int = 300,
        filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
        num_filters_per_size: int = 256,
        p_chunks: int = 8,
        bottleneck_dim: int = 512,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        convs = []
        for k in filter_sizes:
            convs.append(
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=num_filters_per_size,
                    kernel_size=k,
                    padding=k // 2  # keeps length roughly same; similar to Kim
                )
            )
        self.convs = nn.ModuleList(convs)
        self.act = nn.ReLU(inplace=True)
        self.pool = DynamicMaxPool(p_chunks)
        self.dropout = nn.Dropout(dropout)

        total_filters = num_filters_per_size * len(filter_sizes)
        pooled_feat_dim = total_filters * p_chunks

        self.bottleneck = nn.Linear(pooled_feat_dim, bottleneck_dim)
        self.output = nn.Linear(bottleneck_dim, num_labels)
        # Initialize
        nn.init.xavier_uniform_(self.bottleneck.weight)
        nn.init.constant_(self.bottleneck.bias, 0.)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0.)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: (B, T)
        returns logits: (B, L)
        """
        x = self.emb(x_ids)          # (B, T, E)
        x = x.transpose(1, 2)        # (B, E, T) for Conv1d
        pooled_list = []
        for conv in self.convs:
            h = conv(x)              # (B, F, T')
            h = self.act(h)
            hp = self.pool(h)        # (B, F, p)
            pooled_list.append(hp)
        # concat along filter dimension then flatten pooled chunks
        H = torch.cat(pooled_list, dim=1)   # (B, total_filters, p)
        H = H.flatten(1)                    # (B, total_filters * p)
        H = self.dropout(H)
        z = self.act(self.bottleneck(H))    # (B, H_bottleneck)
        z = self.dropout(z)
        logits = self.output(z)             # (B, L)
        return logits

############################################################
# 5) Training / Evaluation
############################################################

@torch.no_grad()
def predict_topk(model: nn.Module, x_ids: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (topk_scores, topk_indices) with shapes (B, k)
    """
    model.eval()
    logits = model(x_ids)
    probs = torch.sigmoid(logits)
    topk_scores, topk_idx = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)
    return topk_scores, topk_idx

def precision_at_k(scores: torch.Tensor, y_true: torch.Tensor, k: int = 5) -> float:
    """
    scores: probabilities or logits (B, L)
    y_true: multi-hot (B, L)
    """
    with torch.no_grad():
        topk_idx = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices
        hits = []
        for i in range(scores.size(0)):
            yset = set((y_true[i] > 0.5).nonzero(as_tuple=True)[0].tolist())
            predk = set(topk_idx[i].tolist())
            if k > 0:
                hits.append(len(yset & predk) / float(k))
        return float(np.mean(hits)) if hits else 0.0

def train_xmlcnn_from_dataframe(
    df: pd.DataFrame,
    data_name: str,
    text_col: str = "text",
    labels_col: str = "labels",
    valid_ratio: float = 0.1,
    min_freq: int = 2,
    max_vocab: int = 100_000,
    max_len: int = 512,
    emb_dim: int = 300,
    filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
    num_filters_per_size: int = 256,
    p_chunks: int = 8,
    bottleneck_dim: int = 512,
    dropout: float = 0.2,
    batch_size: int = 32,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    max_epochs: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 1337,
    
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Build vocab and label maps
    stoi = build_vocab(df[text_col].tolist(), min_freq=min_freq, max_vocab=max_vocab)
    ltoi = build_label_map(df[labels_col].tolist())

    # Split
    idxs = np.arange(len(df))
    np.random.shuffle(idxs)
    n_valid = int(len(df) * valid_ratio)
    valid_idx = idxs[:n_valid]
    train_idx = idxs[n_valid:]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    # Datasets & Loaders
    train_ds = XMTCDataset(df_train, stoi, ltoi, text_col=text_col, labels_col=labels_col, max_len=max_len)
    valid_ds = XMTCDataset(df_valid, stoi, ltoi, text_col=text_col, labels_col=labels_col, max_len=max_len)
    pad_id = stoi.get("<pad>", 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: pad_collate(b, pad_id), num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: pad_collate(b, pad_id), num_workers=0)

    glove_emb_path = '/root/.cache/kagglehub/datasets/thanakomsn/glove6b300dtxt/versions/1/glove.6B.300d.txt'

    emb_matrix = load_glove_embeddings(glove_emb_path, stoi, emb_dim=emb_dim)
    print(f'loading glove embedding models')
    # Model
    model = XMLCNN(
        vocab_size=len(stoi),
        num_labels=len(ltoi),
        emb_dim=emb_dim,
        filter_sizes=filter_sizes,
        num_filters_per_size=num_filters_per_size,
        p_chunks=p_chunks,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
        pad_idx=pad_id,
    ).to(device)
    
    # Initialize embedding weights
    with torch.no_grad():
        model.emb.weight.copy_(emb_matrix)

    model.emb.weight.requires_grad = False

    # Loss/Opt
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train
    best_val = -1.0
    label_weights = None  # placeholder for possible pos_weight usage

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        
        if epoch == 3:
            model.emb.weight.requires_grad = True 

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        all_scores = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                all_scores.append(torch.sigmoid(logits).cpu())
                all_targets.append(yb.cpu())
        val_loss /= max(1, len(valid_ds))
        scores = torch.cat(all_scores, dim=0) if all_scores else torch.empty(0, len(ltoi))
        targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, len(ltoi))
        p1 = precision_at_k(scores, targets, k=1) if len(scores) else 0.0
        p3 = precision_at_k(scores, targets, k=3) if len(scores) else 0.0
        p5 = precision_at_k(scores, targets, k=5) if len(scores) else 0.0

        print(f"Epoch {epoch:02d} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | P@1={p1:.4f} P@3={p3:.4f} P@5={p5:.4f}")

        if p5 > best_val:
            best_val = p5
            torch.save({
                "model_state": model.state_dict(),
                "stoi": stoi, "ltoi": ltoi,
                "config": {
                    "emb_dim": emb_dim,
                    "filter_sizes": filter_sizes,
                    "num_filters_per_size": num_filters_per_size,
                    "p_chunks": p_chunks,
                    "bottleneck_dim": bottleneck_dim,
                    "dropout": dropout,
                    "pad_idx": pad_id
                }
            }, f"xmlcnn_{data_name}_best.pt")
            print("Saved checkpoint -> xmlcnn_best.pt")

    return model, stoi, ltoi

# =========================
# XML-CNN — Evaluate on df_test
# =========================
import torch
import numpy as np
from torch.utils.data import DataLoader

# --- Reuse from your training code ---
# XMTCDataset, pad_collate, XMLCNN, precision_at_k must be defined already.
# If not, import them from your module or paste their definitions here.

@torch.no_grad()
def evaluate_xmlcnn(model, df_test, stoi, ltoi, batch_size=64, max_len=512, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    test_ds = XMTCDataset(df_test, stoi, ltoi, text_col="text", labels_col="labels", max_len=max_len)
    pad_id = stoi.get("<pad>", 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: pad_collate(b, pad_id))

    all_logits, all_targets = [], []
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_targets.append(yb.cpu())

    if not all_logits:
        return {"P@1": 0.0, "P@3": 0.0, "P@5": 0.0, "micro-F1@0.5": 0.0}

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    probs = torch.sigmoid(logits)
    p1 = precision_at_k(probs, targets, k=1)
    p3 = precision_at_k(probs, targets, k=3)
    p5 = precision_at_k(probs, targets, k=5)

    # Micro-F1 at a 0.5 threshold (common quick check)
    y_pred_bin = (probs >= 0.5).int()
    tp = (y_pred_bin & (targets.int())).sum().item()
    fp = (y_pred_bin & (1 - targets.int())).sum().item()
    fn = ((1 - y_pred_bin) & (targets.int())).sum().item()
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    micro_f1 = 2 * prec * rec / (prec + rec + 1e-12)

    return {"P@1": p1, "P@3": p3, "P@5": p5, "micro-F1@0.5": micro_f1}


