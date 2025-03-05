import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
from itertools import product

from process_data import process_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("download_process.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def process_file(file_path: str) -> str:
    """
    ダウンロードしたファイルを加工して必要なデータだけを保存する処理を実装する。
    """
    df = pd.read_csv(file_path)
    dfs = process_data(df)
    processed_dir = file_path.replace("robocup2d_data", "robocup2d_data_processed2")
    os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
    for i, df in enumerate(dfs):
        df.to_csv(processed_dir.replace(".csv", f"_{i}.csv"), index=False)
    return processed_dir


def download_and_process_file(team, file_name, base_url, save_dir):
    """
    指定したURLからファイルをダウンロードし、
    加工処理を実行後、元ファイルを削除する一連の処理
    """
    file_url = f"{base_url}/{team}/{file_name}"
    team_dir = os.path.join(save_dir, team)
    os.makedirs(team_dir, exist_ok=True)
    file_path = os.path.join(team_dir, file_name)

    try:
        with requests.get(file_url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        logging.error(f"Error downloading {file_url}: {e}", exc_info=True)
        return None

    try:
        processed_file = process_file(file_path)
        logging.info(f"Processed {file_path} -> {processed_file}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}", exc_info=True)
        processed_file = None

    try:
        os.remove(file_path)
    except Exception as e:
        logging.error(f"Error deleting {file_path}: {e}", exc_info=True)

    return processed_file


def process_subpath(team, base_url, save_dir, download_num=-1, max_workers=5):
    """
    1つのサブパス（チーム）内で、対象ファイルのダウンロード・加工・削除を並列に処理する。
    ダウンロードするファイル数は download_num で制限可能（-1 の場合は全件）。
    """
    url = f"{base_url}/{team}/"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Error accessing {url}: {e}", exc_info=True)
        return

    soup = BeautifulSoup(response.text, "html.parser")
    files = [
        link["href"]
        for link in soup.find_all("a", href=True)
        if link["href"].endswith("tracking.csv")
    ]
    if download_num != -1:
        files = files[:download_num]

    logging.info(f"Team {team}: {len(files)} files found.")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_and_process_file, team, file_name, base_url, save_dir
            ): file_name
            for file_name in files
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(
                    f"Error in processing a file for team {team}: {e}", exc_info=True
                )
    return results


def process_all_subpaths(base_url, subpaths, save_dir, download_num=-1, max_workers=5):
    """
    サブパスごとに順次、ファイルのダウンロード・加工・削除を実施する。
    HDDの容量制限を考慮し、1サブパス単位で処理を完結させる。
    """
    for team in subpaths:
        logging.info(f"Starting processing team: {team}")
        process_subpath(team, base_url, save_dir, download_num, max_workers)
        logging.info(f"Finished processing team: {team}")


teams = [
    "aeteam2024",
    "cyrus2024",
    "fra2024",
    "helios2024",
    "itandroids2024",
    "mars2024",
    "oxsy2024",
    "r2d2",
    "robocin2024",
    "yushan2024",
]

subpaths = {
    f"{team1}-{team2}" for team1, team2 in product(teams, teams) if team1 != team2
}

process_all_subpaths(
    base_url="http://alab.ise.ous.ac.jp/robocupdata/rc2024-roundrobin",
    subpaths=subpaths,
    save_dir="robocup2d_data",
    download_num=-1,
    max_workers=20,
)
