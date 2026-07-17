# Build the puzzle dataset: download the Kaggle base, then scrape any newer
# puzzles from the NYT API. Run this by hand when you want fresh data:
#
#   python data.py
#
# Everything else just reads data/Connections_Data.csv via connections.load_puzzles().

import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import kagglehub
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
KAGGLE_FILE = DATA_DIR / "Connections_Data_Kaggle.csv"  # untouched base download
DATA_FILE = DATA_DIR / "Connections_Data.csv"           # base + scraped, what we read


def _read_csv(path):
    # "NA" is a real answer word (sodium), so don't let pandas read it as missing.
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def download_base():
    """Download the base Connections dataset from Kaggle into data/."""
    path = kagglehub.dataset_download("eric27n/the-new-york-times-connections")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(path) / "Connections_Data.csv", KAGGLE_FILE)
    return KAGGLE_FILE


def update(rebuild=False):
    """Extend the dataset with any puzzles published since its last date.

    Scrapes every missing day between the dataset's last puzzle and yesterday from
    the NYT Connections API. Days that fail to fetch or parse are skipped with a
    warning. Picks up from DATA_FILE if it exists; pass rebuild=True to start over
    from the Kaggle base.
    """
    if not KAGGLE_FILE.exists():
        download_base()

    source = DATA_FILE if DATA_FILE.exists() and not rebuild else KAGGLE_FILE
    df = _read_csv(source)
    df["Puzzle Date"] = pd.to_datetime(df["Puzzle Date"]).dt.date

    last_date = df["Puzzle Date"].max()
    yesterday = date.today() - timedelta(days=1)
    missing = [
        last_date + timedelta(days=offset)
        for offset in range(1, (yesterday - last_date).days + 1)
    ]

    new_rows = []
    for puzzle_date in missing:
        url = f"https://www.nytimes.com/svc/connections/v2/{puzzle_date}.json"
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            results = response.json()

            word_order = {}
            for group_number, group in enumerate(results["categories"]):
                for card in group["cards"]:
                    # Picture puzzles put the answer in the image's alt text.
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

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df = df.drop_duplicates()
    df = df.sort_values(by=["Puzzle Date", "Starting Row", "Starting Column"])
    df.to_csv(DATA_FILE, index=False)
    print(f"Added {len(new_rows) // 16} new puzzles. Saved to {DATA_FILE}")
    return DATA_FILE


if __name__ == "__main__":
    update()
