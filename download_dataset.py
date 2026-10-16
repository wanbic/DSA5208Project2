
import re
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Configuration
COLLECTION_ID = 2279
YEARS = range(2017, 2025)
OUTPUT_DIR = Path("data")

META_API = "https://api-production.data.gov.sg/v2/public/api"
DOWNLOAD_API = "https://api-open.data.gov.sg/v1/public/api"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
retry = Retry(
    total=8,
    backoff_factor=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)
session.mount("https://", HTTPAdapter(max_retries=retry))


def get_json(url, params=None):
    response = session.get(
        url, params=params, timeout=60
    )
    response.raise_for_status()
    result = response.json()

    if result.get("code") != 0:
        raise RuntimeError(
            f"API error: {result.get('errorMsg')}"
        )

    return result["data"]


def get_datasets():
    """Get dataset IDs and identify their years."""

    url = (
        f"{META_API}/collections/"
        f"{COLLECTION_ID}/metadata"
    )

    data = get_json(url)
    ids = data["collectionMetadata"]["childDatasets"]

    datasets = {}

    for dataset_id in ids:
        url = (
            f"{META_API}/datasets/"
            f"{dataset_id}/metadata"
        )

        metadata = get_json(url)
        info = metadata.get("datasetMetadata", metadata)

        name = info.get("name", "")
        print(f"Found: {name} ({dataset_id})")

        # Match a four-digit year in dataset name
        matches = re.findall(
            r"(?<!\d)20(?:1[7-9]|2[0-4])(?!\d)",
            name
        )

        if len(matches) == 1:
            year = int(matches[0])

            if year in datasets:
                raise ValueError(
                    f"Duplicate dataset for {year}"
                )

            datasets[year] = dataset_id

        time.sleep(1)

    missing = set(YEARS) - set(datasets)

    if missing:
        raise RuntimeError(
            f"Missing datasets for years: {sorted(missing)}"
        )

    return datasets


def get_download_url(dataset_id):
    """Initiate and poll for a download URL."""

    endpoint = (
        f"{DOWNLOAD_API}/datasets/{dataset_id}"
    )

    get_json(f"{endpoint}/initiate-download")

    for attempt in range(60):
        time.sleep(6)

        result = get_json(
            f"{endpoint}/poll-download"
        )

        url = result.get("url")

        if url:
            return url

        print(
            f"  Waiting for download URL "
            f"({attempt + 1}/60)"
        )

    raise TimeoutError(
        f"Download URL timeout: {dataset_id}"
    )


def download_file(url, path):
    """Stream download with a progress bar."""

    temp_path = path.with_suffix(".csv.part")

    try:
        with session.get(
            url, stream=True, timeout=(30, 180)
        ) as response:
            response.raise_for_status()

            total = int(
                response.headers.get(
                    "content-length", 0
                )
            )

            with open(temp_path, "wb") as f:
                with tqdm(
                    total=total if total else None,
                    unit="B",
                    unit_scale=True,
                    desc=path.name,
                ) as bar:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

        temp_path.replace(path)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main():
    print("Fetching collection metadata...")

    datasets = get_datasets()

    print("\nDatasets to download:")
    for year in YEARS:
        print(f"{year}: {datasets[year]}")

    failed = []

    for year in YEARS:
        path = OUTPUT_DIR / f"rainfall_{year}.csv"

        if path.exists() and path.stat().st_size > 0:
            print(f"\nSkipping {year}: already downloaded")
            continue

        print(f"\nDownloading {year}...")

        try:
            url = get_download_url(datasets[year])
            download_file(url, path)

            print(
                f"Saved: {path} "
                f"({path.stat().st_size / 1024**2:.1f} MB)"
            )

        except Exception as e:
            print(f"Failed {year}: {e}")
            failed.append(year)

        # Avoid hitting API rate limits
        time.sleep(6)

    print("\nDownload finished.")

    if failed:
        print("Failed years:", failed)
    else:
        print("All 8 years downloaded successfully.")


if __name__ == "__main__":
    main()
