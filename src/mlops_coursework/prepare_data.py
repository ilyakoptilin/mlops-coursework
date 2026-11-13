import math
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/ai4i2020.csv")
PROCESSED_PATH = Path("data/processed/ai4i2020_processed.csv")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Оставляем только показания датчиков и целевую переменную (убираем ID и утечки TWF/HDF/PWF/OSF/RNF)
    rename_map = {
        "Type": "machine_type",
        "Air temperature [K]": "air_temp_k",
        "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "target",
    }
    clean_df = df[list(rename_map.keys())].rename(columns=rename_map).copy()

    # 2. Создаём новые физические признаки
    clean_df["temp_diff_k"] = (clean_df["process_temp_k"] - clean_df["air_temp_k"]).round(2)
    clean_df["power_w"] = (
        clean_df["torque_nm"] * (clean_df["rotational_speed_rpm"] * 2 * math.pi / 60)
    ).round(2)
    clean_df["overstrain_index"] = (
        clean_df["torque_nm"] * clean_df["tool_wear_min"]
    ).round(2)

    return clean_df


def main() -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(RAW_PATH)
    df_processed = prepare_features(df_raw)
    df_processed.to_csv(PROCESSED_PATH, index=False)

    print(f"Готово! Очищенный датасет v2 сохранён в {PROCESSED_PATH}")
    print(f"Размер таблицы: {df_processed.shape[0]} строк, {df_processed.shape[1]} колонок")
    print(f"Доля поломок (target=1): {df_processed['target'].mean():.2%}")
    print("\nПервые 3 строки:")
    print(df_processed.head(3))


if __name__ == "__main__":
    main()
