import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch.nn.functional as F

import json, glob
from datetime import datetime

import numpy as np

# Handling numpy types in json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# 1. Load the Model and Tokenizer
# 'ProsusAI/finbert' is the most popular pre-trained version
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def _build_head_tail_input_ids(text: str) -> list:
    """Build input_ids using fixed head/tail split (128, -382) and pad to 512.

    This enforces the exact split you requested. Returns a list of length 512
    containing token ids (padded with `pad_token_id`).
    """
    tokens = tokenizer.encode(str(text), add_special_tokens=False)
    if len(tokens) > 510:
        head = tokens[:128]
        tail = tokens[-382:]
        input_ids = [tokenizer.cls_token_id] + head + tail + [tokenizer.sep_token_id]
    else:
        input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]

    if len(input_ids) < 512:
        input_ids += [tokenizer.pad_token_id] * (512 - len(input_ids))

    return input_ids


def get_article_score(text: str, device: torch.device = None) -> float:
    """Return a single sentiment score (pos - neg) for one article.

    Uses the fixed 128 / -382 head-tail token split and does not return masks.
    """
    if not text or pd.isna(text):
        return 0.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    input_ids = _build_head_tail_input_ids(text)
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    return float(probs[0] - probs[1])


def get_day_score(summaries: list, device: torch.device = None):
    """Compute day's average score and article count from a list of summaries.

    Returns: [avg_score, count]
    """
    if not summaries:
        return [0.0, 0.0]

    scores = []
    count = 0
    for s in summaries:
        if s:
            scores.append(get_article_score(s, device=device))
            count += 1

    if count == 0:
        return [0.0, 0.0]

    return [sum(scores) / count, count]


def process_summaries_to_scores(read_glob: str, output_file: str, device: torch.device = None):
    """Read all JSON files matching `read_glob`, compute daily scores, and write `output_file`.

    The output JSON maps date (YYYY-MM-DD) -> [avg_score, count]. The function
    follows the same aggregation logic previously in `main` but uses
    `get_day_score` to compute per-day values.
    """
    file_paths = glob.glob(read_glob)
    if not file_paths:
        print(f"No files matched glob: {read_glob}")
        return {}

    final_dataset = {}
    for path in file_paths:
        with open(path, "r") as f:
            data = json.load(f)
            final_dataset |= data

    print(f"Successfully joined {len(file_paths)} files. Total entries: {len(final_dataset)}")

    dates = [datetime.strptime(key, "%Y%m%d%H%M%S") for key in final_dataset.keys()]
    df = pd.DataFrame({
        "date": dates,
        "text": [final_dataset[key]["full_text"] if final_dataset[key]["status"] == "success" else None for key in final_dataset.keys()],
    })
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    daily_summaries_series = df.groupby(df.index.date)["text"].agg(list)

    all_days = pd.date_range(start=daily_summaries_series.index.min(), end=daily_summaries_series.index.max(), freq="D")
    daily_summaries_series = daily_summaries_series.reindex(all_days, fill_value=[])

    # Use provided device or infer
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    daily_news_scores = {}
    for date in all_days:
        summaries = daily_summaries_series.loc[date]
        score, count = get_day_score(summaries, device=device)
        date_str = date.strftime("%Y-%m-%d")
        daily_news_scores[date_str] = [score, int(count)]

    with open(output_file, "w") as f:
        json.dump(daily_news_scores, f, cls=NpEncoder, indent=4)

    print(f"Wrote daily scores to {output_file}")
    return daily_news_scores

def main():
    process_summaries_to_scores("historical/summaries/20*-20*.json", "finBERT_scores.json")

if __name__ == "__main__":
    main()