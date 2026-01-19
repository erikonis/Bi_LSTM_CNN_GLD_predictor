"""FinBERT sentiment helpers: tokenize, score and aggregate article summaries.

Provides small utilities to load the FinBERT model/tokenizer and convert
article text into a single sentiment score. JSON encoder `NpEncoder` helps
serialize numpy types when writing outputs.
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch.nn.functional as F

import json, glob
from datetime import datetime

import numpy as np


class NpEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# Load the pretrained FinBERT tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def _build_head_tail_input_ids(text: str) -> list:
    """Build 512-length token ids using a fixed head/tail split.

    Args:
        text: Article text to tokenize.

    Returns:
        list[int]: Padded list of token ids of length 512.
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
    """Compute a single sentiment score for `text` using FinBERT.

    Args:
        text: Article text string.
        device: Optional torch.device to run the model on.

    Returns:
        float: Score computed as (pos_prob - neg_prob). Returns 0.0 for empty input.
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
    """Aggregate per-article scores into a day-level mean and count.

    Args:
        summaries: List of article summary strings for a single day.
        device: Optional torch.device to run FinBERT on.

    Returns:
        tuple: (average_score (float), count (int)).
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
    """Read JSON summaries matching `read_glob`, compute per-day FinBERT scores, and write a JSON file.

    Args:
        read_glob: Glob pattern to match input JSON summary files.
        output_file: Path to write daily scores JSON mapping date->(score,count).
        device: Optional torch.device to run scoring on.

    Returns:
        dict: Mapping date-string -> [avg_score, count] that was written to disk.
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