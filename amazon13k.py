'''
Experiments for Amazon13k
'''


# pip install pandas
from xml_cnn_model import train_xmlcnn_from_dataframe, evaluate_xmlcnn

from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple

def _collect_label_ids(rows: List[dict], rel_threshold: float) -> List[int]:
    uniq = set()
    for r in rows:
        inds = r.get("target_ind") or []
        vals = r.get("target_rel") or []
        keep = [i for i, v in zip(inds, vals) if float(v) >= rel_threshold]
        uniq.update(keep)
    return sorted(uniq)

def build_xmlcnn_df_from_jsonl(
    jsonl_path: str,
    text_template: str = "{title}\n\n{content}",
    rel_threshold: float = 0.0,
    existing_ltoi: Optional[Dict[int, int]] = None,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Returns:
      df: 2 columns -> text: str, labels: List[int]  (contiguous IDs)
      ltoi: mapping {original_label_id -> contiguous_id}
    If existing_ltoi is provided, it is reused (and unseen labels are dropped).
    """
    # 1) load JSONL
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for k in ["title", "content", "target_ind", "target_rel"]:
                if k not in obj:
                    raise ValueError(f"Missing key '{k}' in record with uid={obj.get('uid')}")
            if len(obj["target_ind"]) != len(obj["target_rel"]):
                raise ValueError(f"target_ind/target_rel length mismatch for uid={obj.get('uid')}")
            rows.append(obj)

    # 2) build or reuse label map (original → contiguous)
    if existing_ltoi is None:
        originals = _collect_label_ids(rows, rel_threshold)
        ltoi = {orig: i for i, orig in enumerate(originals)}
    else:
        ltoi = existing_ltoi

    # 3) produce DataFrame in XML-CNN shape
    texts, labels = [], []
    for r in rows:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        text = text_template.format(title=title, content=content)

        inds = r.get("target_ind") or []
        vals = r.get("target_rel") or []
        kept = [i for i, v in zip(inds, vals) if float(v) >= rel_threshold]
        mapped = [ltoi[i] for i in kept if i in ltoi]  # drop unseen if using existing map

        texts.append(text)
        labels.append(mapped)

    df = pd.DataFrame({"text": texts, "labels": labels})
    return df, ltoi


df_train, ltoi = build_xmlcnn_df_from_jsonl("data/amazon13k/trn_part1.json", rel_threshold=0.0)
df_test,  _ = build_xmlcnn_df_from_jsonl("data/amazon13k/tst_part1.json",  rel_threshold=0.0, existing_ltoi=ltoi)

# Train your XML-CNN

model, stoi, _ = train_xmlcnn_from_dataframe(
    df_train,
    max_len=256,
    emb_dim=300,
    num_filters_per_size=128,
    p_chunks=6,
    bottleneck_dim=256,
    batch_size=16,
    max_epochs=10, 
    data_name='amazon13k'
)

print(f'amazon 13k ___ results')
metrics = evaluate_xmlcnn(model, df_test, stoi, ltoi)
print(metrics)
