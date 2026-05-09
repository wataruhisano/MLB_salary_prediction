"""
model.py
MLB年俸予測プロジェクト - モデル定義・評価ユーティリティ
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

SEED = 42

# ─────────────────────────────────────────
# 特徴量定義
# ─────────────────────────────────────────
NUMERIC_FEATURES: list[str] = [
    "Age",
    "Age_Squared",
    "is_veteran",
    "PA",
    "WAR(bref)",
    "wRC+",
    "ISO",
    "BB%",
    "K%",
    "Def",
    "BsR",
    "career_award_score",
]

CATEGORY_FEATURES: list[str] = ["Team", "Pos_Group"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    DataFrameから特徴量行列Xとターゲット y（log変換済み）を生成する。

    Parameters
    ----------
    df : pd.DataFrame
        前処理済みのモデル用データ

    Returns
    -------
    X : pd.DataFrame  ダミー変数化済みの特徴量行列
    y : pd.Series     log1p変換済みの年俸
    """
    X_raw = df[NUMERIC_FEATURES + CATEGORY_FEATURES]
    X = pd.get_dummies(X_raw, columns=["Team", "Pos_Group"], drop_first=False)
    y = np.log1p(df["target_salary"])
    return X, y


# ─────────────────────────────────────────
# モデルファクトリ
# ─────────────────────────────────────────
def build_random_forest(seed: int = SEED) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
    )


def build_lightgbm(seed: int = SEED) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=seed,
        n_jobs=-1,
    )


def build_xgboost(seed: int = SEED) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
    )


# ─────────────────────────────────────────
# 評価
# ─────────────────────────────────────────
def evaluate_models(
    models: dict[str, object],
    predictions_log: dict[str, np.ndarray],
    y_test_log: pd.Series,
) -> pd.DataFrame:
    """
    複数モデルの予測精度をまとめた DataFrame を返す。

    Parameters
    ----------
    models : dict  モデル名 -> 学習済みモデル（R2計算用）
    predictions_log : dict  モデル名 -> log変換済みの予測値配列
    y_test_log : pd.Series  log変換済みの正解値

    Returns
    -------
    pd.DataFrame : MAE / MAPE / R2 の比較表
    """
    y_actual = np.expm1(y_test_log)
    rows = []
    for name, y_pred_log in predictions_log.items():
        y_pred = np.expm1(y_pred_log)
        rows.append(
            {
                "Model": name,
                "MAE (USD)": mean_absolute_error(y_actual, y_pred),
                "MAPE (%)": mean_absolute_percentage_error(y_actual, y_pred) * 100,
                "R2 Score": r2_score(y_test_log, y_pred_log),
            }
        )

    return pd.DataFrame(rows).sort_values("MAPE (%)")


def build_residuals_df(
    y_test_log: pd.Series,
    y_pred_log: np.ndarray,
    df_original: pd.DataFrame,
) -> pd.DataFrame:
    """
    残差分析用 DataFrame を構築する（選手名・チーム付き）。

    Parameters
    ----------
    y_test_log    : テスト用の正解値（log変換済み）
    y_pred_log    : 予測値（log変換済み）
    df_original   : Name_Clean / Age / Team 列を含む元データ

    Returns
    -------
    pd.DataFrame
    """
    results = pd.DataFrame(
        {
            "Actual": np.expm1(y_test_log),
            "Predicted": np.expm1(y_pred_log),
        }
    )
    results["Residual"] = results["Actual"] - results["Predicted"]
    results["Absolute_Error"] = results["Residual"].abs()

    bins = [0, 2_000_000, 10_000_000, 20_000_000, np.inf]
    labels = ["Under $2M", "$2M–$10M", "$10M–$20M", "Over $20M"]
    results["Salary_Range"] = pd.cut(results["Actual"], bins=bins, labels=labels)

    meta = df_original[["Name_Clean", "Age", "Team"]]
    return results.join(meta)
