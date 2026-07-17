# Downloading, loading, and inspecting the NYT Connections puzzle dataset.

import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import kagglehub
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "Connections_Data_Kaggle.csv"


def download_dataset() -> Path:
    """Download the Connections dataset from Kaggle and store it in data/."""
    path = kagglehub.dataset_download("eric27n/the-new-york-times-connections")
    downloaded_file = Path(path) / "Connections_Data.csv"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloaded_file, DATA_FILE)

    return DATA_FILE


def load_dataset() -> pd.DataFrame:
    """Load the Connections dataset into a DataFrame, downloading it first if missing."""
    if not DATA_FILE.exists():
        download_dataset()
    return pd.read_csv(DATA_FILE)


def show_dataset(n: int | None = None) -> None:
    """Print the dataset, or just the first n rows if n is given."""
    df = load_dataset()
    print(df.head(n) if n is not None else df)


def update_dataset() -> Path:
    """Fetch puzzles published since the dataset's last date and write them to a new CSV.

    Reads the current dataset, scrapes any missing days between the day after its
    last puzzle and yesterday from the NYT Connections API, and saves the combined
    result to Connections_Data_updated.csv rather than overwriting the original.
    """
    df = load_dataset()
    df["Puzzle Date"] = pd.to_datetime(df["Puzzle Date"]).dt.date

    last_date = df["Puzzle Date"].max()
    yesterday = date.today() - timedelta(days=1)
    missing_dates = [
        last_date + timedelta(days=offset)
        for offset in range(1, (yesterday - last_date).days + 1)
    ]

    new_rows = []
    for puzzle_date in missing_dates:
        url = f"https://www.nytimes.com/svc/connections/v2/{puzzle_date}.json"
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            results = response.json()

            word_order = {}
            for group_number, group in enumerate(results["categories"]):
                for card in group["cards"]:
                    word = card.get("content", card.get("image_alt_text"))
                    word_order[int(card["position"])] = (word, group["title"], group_number)

            for position in range(16):
                word, group_name, group_level = word_order[position]
                new_rows.append({
                    "Game ID": results["id"],
                    "Puzzle Date": puzzle_date,
                    "Word": word,
                    "Group Name": group_name,
                    "Group Level": group_level,
                    "Starting Row": (position // 4) + 1,
                    "Starting Column": (position % 4) + 1,
                })
        except Exception as e:
            print(f"Skipping {puzzle_date}: {e}")

        time.sleep(0.2)

    output_file = DATA_DIR / "Connections_Data.csv"

    if not new_rows:
        print("No new puzzles to add.")
        return output_file

    updated_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    updated_df = updated_df.drop_duplicates()
    updated_df = updated_df.sort_values(by=["Puzzle Date", "Starting Row", "Starting Column"])

    date_counts = updated_df["Puzzle Date"].value_counts()
    group_counts = updated_df.groupby(["Puzzle Date", "Group Name"]).size()
    bad_dates = date_counts[date_counts != 16]
    bad_groups = group_counts[group_counts != 4]
    if len(bad_dates) or len(bad_groups):
        print(f"Warning: inconsistent row counts.\nDates:\n{bad_dates}\nGroups:\n{bad_groups}")

    updated_df.to_csv(output_file, index=False)
    print(f"Added {len(new_rows) // 16} new puzzles. Saved to {output_file}")
    return output_file

update_dataset()
