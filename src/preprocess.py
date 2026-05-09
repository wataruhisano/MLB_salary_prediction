"""
preprocess.py
MLB年俸予測プロジェクト - データ読み込み・クリーニング・特徴量エンジニアリング
"""

import os
import re
import shutil

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # .env を自動読み込み


# ─────────────────────────────────────────
# 定数
# ─────────────────────────────────────────
SEED = 42
MIN_SALARY_THRESHOLD = 700_000
YEARS = [2017, 2018, 2019, 2021, 2022, 2023, 2024]

# 環境に応じてパスを切り替え
# Colab: /content/mlb_analytics.duckdb
# ローカル: .env の DB_PATH（未設定なら ./mlb_analytics.duckdb）
LOCAL_DB_PATH = os.getenv("DB_PATH", "./mlb_analytics.duckdb")


def get_data_dir() -> str:
    """
    データディレクトリを返す。
    - ローカル（VS Code）: .env の DATA_DIR
    - Google Colab      : 環境変数 COLAB_DATA_DIR（または引数で上書き）
    """
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise EnvironmentError(
            "DATA_DIR が設定されていません。\n"
            ".env.example を参考に .env を作成してください。"
        )
    return data_dir


# ─────────────────────────────────────────
# クリーニング関数
# ─────────────────────────────────────────
def clean_salary(x) -> float:
    """'$5,850,000' のような文字列を数値に変換する"""
    if isinstance(x, str):
        return float(re.sub(r"[$,]", "", x))
    return x


def clean_name(x) -> str:
    """'Matt Wallner*' のような名前末尾の記号を除去する"""
    if isinstance(x, str):
        return re.sub(r"[*#]", "", x).strip()
    return x


def categorize_pos_bref(pos: str) -> str:
    """Baseball Reference のポジション表記をグループ化する"""
    pos = str(pos).upper()
    if "2" in pos:
        return "Catcher"
    if any(n in pos for n in ["3", "4", "5", "6"]):
        return "Infield"
    if any(n in pos for n in ["7", "8", "9"]):
        return "Outfield"
    if "D" in pos:
        return "DH"
    return "Other"


# ─────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────
def load_salary(con: duckdb.DuckDBPyConnection, data_dir: str) -> None:
    """年俸CSVを読み込んでDuckDBへ格納する"""
    salary_csv = os.path.join(data_dir, "mlb_salary_data.csv")
    if not os.path.exists(salary_csv):
        raise FileNotFoundError(f"Salary CSV が見つかりません: {salary_csv}")

    df = pd.read_csv(salary_csv, encoding="latin1")
    df["Salary_Numeric"] = df["Salary"].apply(clean_salary)
    df["Name_Clean"] = df["Name"].apply(clean_name)
    con.execute("CREATE OR REPLACE TABLE t_salary AS SELECT * FROM df")
    print(f"[OK] Salary data loaded: {len(df)} rows")


def load_bref_stats(
    con: duckdb.DuckDBPyConnection, data_dir: str, years: list[int] = YEARS
) -> None:
    """Baseball Reference の年度別成績CSVを読み込んでDuckDBへ格納する"""
    frames = []
    for year in years:
        path = os.path.join(data_dir, f"{year}_stats.csv")
        if not os.path.exists(path):
            print(f"[SKIP] {year}_stats.csv が見つかりません")
            continue
        df = pd.read_csv(path)
        df["Year"] = year
        name_col = "Name" if "Name" in df.columns else df.columns[1]
        df["Name_Clean"] = df[name_col].apply(clean_name)
        frames.append(df)
        print(f"[OK] BRef {year} loaded")

    if not frames:
        raise RuntimeError("BRef データが1件も読み込めませんでした")

    full_df = pd.concat(frames, ignore_index=True)
    con.execute("CREATE OR REPLACE TABLE t_stats AS SELECT * FROM full_df")


def load_fangraphs_stats(
    con: duckdb.DuckDBPyConnection, data_dir: str, years: list[int] = YEARS
) -> None:
    """FanGraphs の年度別成績CSVを読み込んでDuckDBへ格納する"""
    frames = []
    for year in years:
        path = os.path.join(data_dir, f"fg_stats_{year}.csv")
        if not os.path.exists(path):
            print(f"[SKIP] fg_stats_{year}.csv が見つかりません")
            continue
        df = pd.read_csv(path)
        df["Year"] = year
        if "Name" in df.columns:
            df["Name_Clean"] = df["Name"].apply(clean_name)
        else:
            df["Name_Clean"] = (
                df.iloc[:, 0].str.replace(r"^\d+", "", regex=True).apply(clean_name)
            )
        frames.append(df)
        print(f"[OK] FanGraphs {year} loaded")

    if not frames:
        raise RuntimeError("FanGraphs データが1件も読み込めませんでした")

    full_df = pd.concat(frames, ignore_index=True)
    con.execute("CREATE OR REPLACE TABLE t_fg_stats AS SELECT * FROM full_df")


# ─────────────────────────────────────────
# データ結合・特徴量エンジニアリング
# ─────────────────────────────────────────
MERGE_QUERY = """
WITH merged_data AS (
    SELECT
        s.*,
        s.Age * s.Age AS Age_Squared,
        ah.mvp_5yr,
        ah.ss_5yr,
        ah.gg_5yr,
        ah.as_5yr,
        (COALESCE(ah.mvp_5yr, 0) * 10 + COALESCE(ah.ss_5yr, 0) * 3 +
         COALESCE(ah.gg_5yr, 0) * 2 + COALESCE(ah.as_5yr, 0) * 1) AS career_award_score,
        f.wOBA, f.xwOBA, f."wRC+", f.BsR, f.Off, f.Def, f.ISO, f.BABIP,
        f.AVG AS f_AVG, f.OBP AS f_OBP, f.SLG AS f_SLG,
        f."BB%" AS "BB%", f."K%" AS "K%",
        f.WAR AS fWAR, f.HR AS f_HR,
        y.Salary_Numeric AS target_salary
    FROM t_stats s
    JOIN t_fg_stats f
      ON trim(LOWER(CAST(s.Name_Clean AS VARCHAR))) = trim(LOWER(CAST(f.Name_Clean AS VARCHAR)))
     AND s.Year = f.Year
    JOIN t_salary y
      ON trim(LOWER(CAST(s.Name_Clean AS VARCHAR))) = trim(LOWER(CAST(y.Name_Clean AS VARCHAR)))
     AND s.Year = y.Year
    LEFT JOIN t_career_awards ah
      ON trim(LOWER(CAST(s.Name_Clean AS VARCHAR))) = trim(LOWER(CAST(ah.Name_Clean AS VARCHAR)))
     AND s.Year = ah.Year
    WHERE y.Salary_Numeric > 0
)
SELECT * EXCLUDE(rn) FROM (
    SELECT *,
           ROW_NUMBER() OVER(PARTITION BY Name_Clean, Year ORDER BY PA DESC) AS rn
    FROM merged_data
)
WHERE rn = 1
ORDER BY Year, Name_Clean
"""


def build_model_df(
    con: duckdb.DuckDBPyConnection, min_salary: int = MIN_SALARY_THRESHOLD
) -> pd.DataFrame:
    """BRef・FanGraphs・年俸を結合してモデル用DataFrameを構築する"""
    df = con.execute(MERGE_QUERY).df()

    if "WAR" in df.columns:
        df = df.rename(columns={"WAR": "WAR(bref)"})
    if "fWAR" in df.columns:
        df = df.rename(columns={"fWAR": "WAR(fg)"})

    df = df.loc[:, ~df.columns.duplicated()]
    df.drop(columns=[c for c in ["Player", "Rk"] if c in df.columns], inplace=True)

    award_cols = ["mvp_5yr", "ss_5yr", "gg_5yr", "as_5yr", "career_award_score"]
    df[award_cols] = df[award_cols].fillna(0)

    df = df[df["target_salary"] >= min_salary].copy()
    df["is_veteran"] = (df["Age"] >= 28).astype(int)
    df["Pos_Group"] = df["Pos"].apply(categorize_pos_bref)

    print(f"[OK] モデル用データ: {len(df)} 行")
    return df


# ─────────────────────────────────────────
# DuckDB ユーティリティ
# ─────────────────────────────────────────
def init_db(data_dir: str | None = None) -> duckdb.DuckDBPyConnection:
    """
    DuckDB に接続する。
    - ローカル: LOCAL_DB_PATH（.env の DB_PATH）に新規作成
    - Colab  : data_dir に既存DBがあればコピーして再利用
    """
    if data_dir:
        drive_db = os.path.join(data_dir, "mlb_analytics.duckdb")
        if os.path.exists(drive_db):
            shutil.copy(drive_db, LOCAL_DB_PATH)
            print("[OK] 既存 DuckDB をコピーしました")
    return duckdb.connect(LOCAL_DB_PATH)


def sync_db(con: duckdb.DuckDBPyConnection, data_dir: str | None = None) -> None:
    """
    DuckDB を閉じる。
    Colab の場合は data_dir へコピーして Drive に同期する。
    """
    con.close()
    if data_dir:
        dest = os.path.join(data_dir, "mlb_analytics.duckdb")
        shutil.copy(LOCAL_DB_PATH, dest)
        print(f"[OK] DuckDB を同期しました → {dest}")
    else:
        print(f"[OK] DuckDB を保存しました → {LOCAL_DB_PATH}")
