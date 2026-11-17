import urllib.request
from pathlib import Path

from mlops_coursework.features import load_config

FALLBACK_URLS = [
    "[https://raw.githubusercontent.com/sharmaroshan/Predictive-Maintenance-Using-Machine-Learning/master/ai4i2020.csv](https://raw.githubusercontent.com/sharmaroshan/Predictive-Maintenance-Using-Machine-Learning/master/ai4i2020.csv)",
]


def _download_file(url: str, destination: Path) -> None:
    clean_url = url.strip().strip("[]()")
    req = urllib.request.Request(clean_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response, destination.open("wb") as out:
        out.write(response.read())


def main() -> None:
    cfg = load_config()["train"]
    raw_path = Path(cfg["raw_data_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    urls = [cfg["data_url"], *FALLBACK_URLS]
    last_error: Exception | None = None
    for url in urls:
        try:
            _download_file(url, raw_path)
            if raw_path.stat().st_size > 100_000:
                print(f"Downloaded dataset to {raw_path}")
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"Failed to download dataset from all sources: {last_error}")


if __name__ == "__main__":
    main()
