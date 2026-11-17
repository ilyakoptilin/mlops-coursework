from pathlib import Path

import pandas as pd

from mlops_coursework.features import load_config, prepare_raw_dataframe


def main() -> None:
    cfg = load_config()["train"]
    raw_path = Path(cfg["raw_data_path"])
    processed_path = Path(cfg["processed_data_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path)
    processed_df = prepare_raw_dataframe(raw_df)
    processed_df.to_csv(processed_path, index=False)
    print(f"Saved processed dataset ({len(processed_df)} rows) to {processed_path}")


if __name__ == "__main__":
    main()
