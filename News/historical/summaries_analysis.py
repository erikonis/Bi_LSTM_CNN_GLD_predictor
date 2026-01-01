import subprocess
import pandas as pd
from datetime import datetime
import json
import dtale

# Your raw data dictionary
import json
import glob

# List all your json files (e.g., news_2015.json, news_2016.json...)
file_paths = glob.glob("summaries/20*-20*.json")

final_dataset = {}

for path in file_paths:
    with open(path, 'r') as f:
        data = json.load(f)
        # The '|=' operator merges 'data' into 'final_dataset' in-place
        final_dataset |= data

print(f"Successfully joined {len(file_paths)} files. Total entries: {len(final_dataset)}")

# 1. Convert keys to datetime objects
# GDELT keys are usually YYYYMMDDHHMMSS
dates = [datetime.strptime(key, "%Y%m%d%H%M%S") for key in final_dataset.keys()]
success_dates = [datetime.strptime(key, "%Y%m%d%H%M%S") for key in final_dataset.keys() if final_dataset[key]['status'] == 'success']

print(f"Total entries: {len(dates)}")
print(f"Successful entries: {len(success_dates)}")

# 2. Create a basic DataFrame with a dummy 'count' column
df = pd.DataFrame({'date': dates, 'count': 1})
df_succ = pd.DataFrame({'date': success_dates, 'count': 1})



# 3. Set the date as the index
df.set_index('date', inplace=True)
df_succ.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)
df_succ.index = pd.to_datetime(df_succ.index)

# 4. Resample to Daily ('D') and sum the counts
# This will automatically insert rows for missing days and fill with 0
daily_counts = df.resample('D').sum().fillna(0).astype(int)
daily_counts_succ = df_succ.resample('D').sum().fillna(0).astype(int)

zero_days_count = (daily_counts['count'] == 0).sum()
no_success_days_count = (daily_counts_succ['count'] == 0).sum()

# ======= STATS By year =========
# 1. Calculate Article Counts (Total and Successful) per Year
# This replaces your long total_entries and successful_entries dictionaries
yearly_totals = daily_counts['count'].resample('YE').sum()
yearly_success = daily_counts_succ['count'].resample('YE').sum()

# 2. Calculate Failed Entries
yearly_failed = yearly_totals - yearly_success

# 3. Handle Archived Entries using the final_dataset
# We convert the final_dataset to a temporary DataFrame to make grouping easy
df_raw = pd.DataFrame.from_dict(final_dataset, orient='index')
df_raw.index = pd.to_datetime(df_raw.index, format="%Y%m%d%H%M%S")

# Filter for archived successes and group by year
archived_mask = (df_raw['status'] == 'success') & (df_raw['source_type'] == 'archived')
yearly_archived = df_raw[archived_mask].resample('YE').size().reindex(yearly_totals.index, fill_value=0)

# 4. Calculate Zero Day Stats (from previous logic)
yearly_empty_days = (daily_counts['count'] == 0).resample('YE').sum()
yearly_unsucc_days = (daily_counts_succ['count'] == 0).resample('YE').sum()

# 5. Build the Final Stats Table
stats_comparison = pd.DataFrame({
    'Total Articles': yearly_totals,
    'Successful Fetches': yearly_success,
    'Failed Fetches': yearly_failed,
    'Archived (Wayback)': yearly_archived,
    'Days with 0 News': yearly_empty_days,
    'Days without successful fetch' : yearly_unsucc_days
})

# Clean up: Make index just the Year number
stats_comparison.index = stats_comparison.index.year

# d = dtale.show(stats_comparison, subprocess=False)
print(stats_comparison)
print("------- Overall Summary Statistics ------")
print(daily_counts.describe())
print(daily_counts_succ.describe())
# d.open_browser()