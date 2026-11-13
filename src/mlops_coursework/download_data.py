from pathlib import Path
import urllib.request
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
RAW_PATH = Path("data/raw/ai4i2020.csv")


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю датасет из {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, RAW_PATH)

    df = pd.read_csv(RAW_PATH)
    print(f"Готово! Файл сохранён в {RAW_PATH}")
    print(f"Размер таблицы: {df.shape[0]} строк, {df.shape[1]} колонок")
    print("\nПервые 3 строки:")
    print(df.head(3))


if __name__ == "__main__":
    main()
