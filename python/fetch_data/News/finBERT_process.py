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

def get_finbert_summary_sentiment(text):
    """
    Processes text and returns sentiment probabilities. Takes the summary (beginning of the text only).
    :param text: input text string
    :return: list of probabilities [positive_prob, negative_prob, neutral_prob]
    """
    if not text or pd.isna(text):
        return [0.0, 0.0, 1.0]  # Return 'Neutral' for empty text

    # Tokenize (FinBERT has a 512 token limit)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Perform Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert raw logits to probabilities (0 to 1 range)
    probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
    
    # FinBERT order: [Positive, Negative, Neutral]
    return probs.tolist()

def get_head_tail_sentiment(text):
    """
    Processes text by taking head and tail tokens to fit BERT limit.
    (aiming at summary and conclusion parts of the text).

    Would be better for longer articles.

    :param text: input text string
    :return: list of probabilities [positive_prob, negative_prob, neutral_prob]
    """

    if not text or pd.isna(text):
        return [0.0, 0.0, 1.0]  # Return 'Neutral' for empty text

    # 1. Tokenize everything (do not truncate yet)
    # We add add_special_tokens=False because we will manually add them
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # 2. Slice the tokens (BERT limit is 512, but we need 2 spots for [CLS] and [SEP])
    # We take the first 255 and the last 255
    if len(tokens) > 510:
        head = tokens[:255]
        tail = tokens[-255:]
        # Combine: [CLS] + head + tail + [SEP]
        input_ids = [tokenizer.cls_token_id] + head + tail + [tokenizer.sep_token_id]
    else:
        # If it's already short enough, just add the special tokens back
        input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]

    # 3. Convert to tensor
    input_tensor = torch.tensor([input_ids])
    
    # 4. Run Model
    with torch.no_grad():
        outputs = model(input_tensor)
        # FinBERT order is typically [Positive, Negative, Neutral]
        probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
    
    return probs.tolist()

def get_day_sentiment(summaries: list):
    """
    Aggregates daily sentiment from a list of summaries.
    :param summaries: list of text summaries for the day
    :return: [positive_prob, negative_prob, neutral_prob, number_of_articles] averaged over all summaries.
    """
    if not summaries:
        return [0.0, 0.0, 0.0, 0.0]  # pos, neg, neu, count

    total_probs = [0.0, 0.0, 0.0]
    count = 0

    for summary in summaries:
        probs = get_head_tail_sentiment(summary)
        total_probs[0] += probs[0]
        total_probs[1] += probs[1]
        total_probs[2] += probs[2]
        count += 1

    # Average the probabilities
    avg_probs = [p / count for p in total_probs]
    avg_probs.append(count)
    return avg_probs

def get_day_score(summaries: list):
    """
    Computes a single sentiment score for the day from summaries.
    :param summaries: list of text summaries for the day
    :param single_score: if True, returns a single reduced score; else returns full probabilities
    :return: [single_score, number_of_articles]
    """
    daily_probs = get_day_sentiment(summaries)
    return [daily_probs[0] - daily_probs[1], daily_probs[3]]  # single score, count

def batch_head_tail_tokenize(texts: list, device : torch.device = torch.device("cpu")):    
    """
    Tokenizes a batch of texts using head-tail strategy for BERT.
    512 token limit handled by taking first 255 and last 255 tokens.

    :param texts: list of text strings
    :type texts: list[str]
    :param device: torch device to place tensors on
    :type device: torch.device
    :return: dict with 'input_ids' and 'attention_mask' tensors
    """
    batch_input_ids = []
    batch_attention_masks = []
    
    for text in texts:
        # 1. Encode without truncation/padding to get raw tokens
        tokens = tokenizer.encode(str(text), add_special_tokens=False, verbose=False)
        
        # 2. Apply Head + Tail (255 tokens each = 510 + 2 special = 512)
        if len(tokens) > 510:
            head = tokens[:128]
            tail = tokens[-382:]
            input_ids = [tokenizer.cls_token_id] + head + tail + [tokenizer.sep_token_id]
        else:
            input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
            # Pad manually to max_len
            input_ids += [tokenizer.pad_token_id] * (512 - len(input_ids))
            
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [1 if tid != tokenizer.pad_token_id else 0 for tid in input_ids]
        
        batch_input_ids.append(input_ids)
        batch_attention_masks.append(mask)
        
    return {
        'input_ids': torch.tensor(batch_input_ids).to(device),
        'attention_mask': torch.tensor(batch_attention_masks).to(device)
    }

#TODO: Extract the method from main

def main():
    # List all your json files (e.g., news_2015.json, news_2016.json...)
    file_paths = glob.glob("historical/summaries/20*-20*.json")

    final_dataset = {}

    #Load the files and merge them
    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            final_dataset |= data

    print(f"Successfully joined {len(file_paths)} files. Total entries: {len(final_dataset)}")

    dates = [datetime.strptime(key, "%Y%m%d%H%M%S") for key in final_dataset.keys()]
    df = pd.DataFrame({'date': dates, 'text': [final_dataset[key]['full_text'] if final_dataset[key]['status'] == 'success' else None for key in final_dataset.keys()]})
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)

    daily_summaries_series = df.groupby(df.index.date)['text'].agg(list)
    
    # Getting all the days in the range
    all_days = pd.date_range(start=daily_summaries_series.index.min(), 
                             end=daily_summaries_series.index.max(), 
                             freq='D')
    # Fill missing dates with empty lists
    daily_summaries_series = daily_summaries_series.reindex(all_days, fill_value=[])

    # 1. Flatten data for batch processing
    flat_data = []
    for date, summaries in daily_summaries_series.items():
        for s in summaries:
            if s: flat_data.append((date, s))

    # 2. Process in batches
    batch_size = 16 
    results_map = {} # date -> list of scores

    print(f"Starting FinBERT batch processing for {len(flat_data)} articles...")
    
    for i in range(0, len(flat_data), batch_size):
        batch = flat_data[i : i + batch_size]

        batch_dates = [item[0] for item in batch]
        batch_texts = [item[1] for item in batch]
        
        # Load model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Tokenize and predict
        inputs = batch_head_tail_tokenize(batch_texts, device=device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            
        # Store results
        for j, prob_vec in enumerate(probs):
            # Calculate single score: Pos - Neg
            score = prob_vec[0] - prob_vec[1]
            date_key = batch_dates[j].strftime("%Y-%m-%d")
            
            if date_key not in results_map:
                results_map[date_key] = []
            results_map[date_key].append(score)

    # 3. Final Aggregation (Average scores per day)
    daily_news_scores = {}
    for date in all_days: # using your 'all_days' range from before
        date_str = date.strftime("%Y-%m-%d")
        scores = results_map.get(date_str, [])
        
        if not scores:
            daily_news_scores[date_str] = [0.0, 0.0] # Missing day
        else:
            count = len(scores)
            daily_news_scores[date_str] = [sum(scores) / count, count]

    with open("finBERT_scores.json", "w") as f:
        json.dump(daily_news_scores, f, cls=NpEncoder, indent=4)

if __name__ == "__main__":
    main()