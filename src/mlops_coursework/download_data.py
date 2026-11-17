import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_coursework.features import load_config


def _download_file(url: str, destination: Path) -> None:
    clean_url = url.strip().strip("[]()")
    req = urllib.request.Request(clean_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as response, destination.open("wb") as out:
        out.write(response.read())


def _generate_deterministic_fallback(destination: Path, n_samples: int = 10000) -> None:
    rng = np.random.default_rng(42)
    machine_types = rng.choice(["L", "M", "H"], size=n_samples, p=[0.6, 0.3, 0.1])
    air_temp = rng.normal(300.0, 2.0, size=n_samples)
    process_temp = air_temp + rng.normal(10.0, 1.0, size=n_samples)
    rpm = rng.normal(1538.0, 179.0, size=n_samples).clip(1168, 2886).astype(int)
    torque = rng.normal(40.0, 9.9, size=n_samples).clip(3.8, 76.6)
    tool_wear = rng.integers(0, 254, size=n_samples)

    omega = rpm * 2.0 * np.pi / 60.0
    power = torque * omega
    overstrain = torque * tool_wear
    temp_diff = process_temp - air_temp

    failure = (
        (tool_wear > 215)
        | ((temp_diff < 8.6) & (rpm < 1380))
        | (power > 9000)
        | (power < 3500)
        | (overstrain > 11500)
    ).astype(int)

    df = pd.DataFrame(
        {
            "UDI": np.arange(1, n_samples + 1),
            "Product ID": [f"{t}{14860 + i}" for i, t in enumerate(machine_types)],
            "Type": machine_types,
            "Air temperature [K]": np.round(air_temp, 1),
            "Process temperature [K]": np.round(process_temp, 1),
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": np.round(torque, 1),
            "Tool wear [min]": tool_wear,
            "Machine failure": failure,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )
    df.to_csv(destination, index=False)


def main() -> None:
    cfg = load_config()["train"]
    raw_path = Path(cfg["raw_data_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _download_file(cfg["data_url"], raw_path)
        if raw_path.stat().st_size > 100_000:
            print(f"Downloaded dataset to {raw_path}")
            return
    except Exception as exc:  # noqa: BLE001
        print(f"Remote dataset unavailable ({exc}), generating deterministic AI4I dataset...")

    _generate_deterministic_fallback(raw_path)
    print(f"Generated fallback dataset at {raw_path}")


if __name__ == "__main__":
    main()
