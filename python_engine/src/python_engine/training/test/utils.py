from pathlib import Path
import pandas as pd


def preprocessing(filename: Path):
    df = pd.read_csv(str(filename))

    new_df = pd.DataFrame({"Date": df.iloc[:, 0],
                           "Open": df.iloc[:, 1],
                           "High": df.iloc[:, 2],
                           "Low": df.iloc[:, 3],
                           "Close": df.iloc[:, 4],
                           "Volume": df.iloc[:, 5]})

    up_and_down = [0.]
    changes = [0.]

    for idx in range(1, len(new_df)):
        up_and_down.append(new_df.iloc[idx].Close - new_df.iloc[idx - 1].Close)
        ch = ((new_df.iloc[idx].Close - new_df.iloc[idx - 1].Close) / new_df.iloc[idx - 1].Close) * 100
        changes.append(ch)

    new_df["Ups and Downs"] = up_and_down
    new_df["Changes"] = changes

    new_df.to_csv(str(filename.parent.joinpath(filename.stem + "_prep.csv")), index=False)