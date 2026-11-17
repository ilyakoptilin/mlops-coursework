import urllib.request
from pathlib import Path

from mlops_coursework.features import load_config


def main() -> None:
    cfg = load_config()["train"]
    raw_path = Path(cfg["raw_data_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(cfg["data_url"], raw_path)
    print(f"Downloaded dataset to {raw_path}")


if __name__ == "__main__":
    main()
