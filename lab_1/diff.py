import pandas as pd
import numpy as np
import sys
import os

def compare_csv(file1, file2, output=None):
    for f in [file1, file2]:
        if not os.path.exists(f):
            print(f"Ошибка: файл {f} не найден.")
            return
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if not df1.columns.equals(df2.columns):
        print("Предупреждение: столбцы файлов различаются.")
        print("Файл 1:", df1.columns.tolist())
        print("Файл 2:", df2.columns.tolist())
        common_cols = df1.columns.intersection(df2.columns)
        if len(common_cols) == 0:
            print("Нет общих столбцов. Выход.")
            return
        df1 = df1[common_cols]
        df2 = df2[common_cols]
        print("Будут использованы общие столбцы:", common_cols.tolist())

    if len(df1) != len(df2):
        print(f"Предупреждение: разное число строк: {len(df1)} vs {len(df2)}. Будет выполнено сравнение по {min(len(df1), len(df2))} первым строкам.")
        min_len = min(len(df1), len(df2))
        df1 = df1.iloc[:min_len]
        df2 = df2.iloc[:min_len]

    time_col = df1.columns[0]
    data_cols = df1.columns[1:]
    eps = 1e-12
    results = {}

    for col in data_cols:
        v1 = df1[col].values.astype(float)
        v2 = df2[col].values.astype(float)

        denom = np.abs(v1) + np.abs(v2) + eps
        rel_diff = 200.0 * np.abs(v1 - v2) / denom
        rel_diff = np.nan_to_num(rel_diff)

        mean_diff = np.mean(rel_diff)
        max_diff = np.max(rel_diff)
        std_diff = np.std(rel_diff)

        results[col] = {
            'mean_%': mean_diff,
            'max_%': max_diff,
            'std_%': std_diff
        }

    print(f"Сравнение файлов: {os.path.basename(file1)} и {os.path.basename(file2)}")
    print("-"*60)
    for col, stats in results.items():
        print(f"{col:25} | среднее = {stats['mean_%']:8.3f}%")

    if output:
        diff_df = pd.DataFrame()
        diff_df[time_col] = df1[time_col]
        for col in data_cols:
            v1 = df1[col].values
            v2 = df2[col].values
            denom = np.abs(v1) + np.abs(v2) + eps
            rel_diff = 200.0 * np.abs(v1 - v2) / denom
            diff_df[f'{col}_diff%'] = rel_diff
        diff_df.to_csv(output, index=False)
        print(f"Построчные различия сохранены в {output}")

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    compare_csv(file1, file2, output)
