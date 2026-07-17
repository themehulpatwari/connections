# Downloading, loading, and inspecting the NYT Connections puzzle dataset.

import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import kagglehub
import pandas as pd
import requests

from src.puzzle import Group, Puzzle

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The Kaggle download is the base; update_dataset() extends it with freshly
# scraped puzzles to produce DATA_FILE, which everything downstream reads.
KAGGLE_FILE = DATA_DIR / "Connections_Data_Kaggle.csv"
DATA_FILE = DATA_DIR / "Connections_Data.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    # The puzzle word "NA" is a real answer (a periodic table symbol), so default
    # NaN parsing would silently turn it into a missing value.
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def download_dataset() -> Path:
    """Download the base Connections dataset from Kaggle and store it in data/."""
    path = kagglehub.dataset_download("eric27n/the-new-york-times-connections")
    downloaded_file = Path(path) / "Connections_Data.csv"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloaded_file, KAGGLE_FILE)

    return KAGGLE_FILE


def update_dataset(rebuild: bool = False) -> Path:
    """Extend the dataset with puzzles published since its last date.

    Scrapes every missing day between the dataset's last puzzle and yesterday
    from the NYT Connections API, then writes the combined result to DATA_FILE.
    Days that fail to fetch or parse are skipped with a warning rather than
    aborting the run.

    Picks up from DATA_FILE when it exists so a routine run only fetches the new
    days; pass rebuild=True to discard it and start over from the Kaggle base.
    """
    if not KAGGLE_FILE.exists():
        download_dataset()

    source = DATA_FILE if DATA_FILE.exists() and not rebuild else KAGGLE_FILE
    df = _read_csv(source)
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

    date_counts = df["Puzzle Date"].value_counts()
    group_counts = df.groupby(["Puzzle Date", "Group Name"]).size()
    bad_dates = date_counts[date_counts != 16]
    bad_groups = group_counts[group_counts != 4]
    blank_words = df[df["Word"].str.strip() == ""]
    if len(bad_dates) or len(bad_groups):
        print(f"Warning: inconsistent row counts.\nDates:\n{bad_dates}\nGroups:\n{bad_groups}")
    if len(blank_words):
        print(f"Warning: {len(blank_words)} rows have a blank word.\n{blank_words}")

    df.to_csv(DATA_FILE, index=False)
    print(f"Added {len(new_rows) // 16} new puzzles. Saved to {DATA_FILE}")
    return DATA_FILE


def load_dataset() -> pd.DataFrame:
    """Load the canonical dataset into a DataFrame, building it first if missing."""
    if not DATA_FILE.exists():
        update_dataset()
    return _read_csv(DATA_FILE)


def load_puzzles() -> list[Puzzle]:
    """Load the canonical dataset as Puzzle objects, one per game, oldest first."""
    df = load_dataset()
    df["Puzzle Date"] = pd.to_datetime(df["Puzzle Date"]).dt.date

    puzzles = []
    for (puzzle_date, game_id), rows in df.groupby(["Puzzle Date", "Game ID"], sort=True):
        rows = rows.sort_values(["Starting Row", "Starting Column"])
        groups = tuple(
            Group(
                name=level_rows["Group Name"].iloc[0],
                level=int(level),
                words=tuple(level_rows["Word"]),
            )
            for level, level_rows in rows.groupby("Group Level", sort=True)
        )
        puzzles.append(
            Puzzle(
                game_id=int(game_id),
                date=puzzle_date,
                words=tuple(rows["Word"]),
                groups=groups,
            )
        )
    return puzzles


def show_dataset(n: int | None = None) -> None:
    """Print the dataset, or just the first n rows if n is given."""
    df = load_dataset()
    print(df.head(n) if n is not None else df)
