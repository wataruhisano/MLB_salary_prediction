"""
train.py
MLB年俸予測プロジェクト - 学習・評価・SHAP分析のエントリーポイント

【VS Code / ローカル】
    python src/train.py
    ※ .env に DATA_DIR を設定しておくこと

【Google Colab】
    import sys; sys.path.append('/content/mlb_salary_prediction/src')
    from train import run
    run(data_dir="/content/drive/My Drive/個人開発/MLB_DATA_PREDICT/data/")
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from preprocess import (
    build_model_df,
    get_data_dir,
    init_db,
    load_bref_stats,
    load_fangraphs_stats,
    load_salary,
    sync_db,
)
from model import (
    build_features,
    build_lightgbm,
    build_random_forest,
    build_xgboost,
    build_residuals_df,
    evaluate_models,
)

load_dotenv()

SEED = 42
TEST_SIZE = 0.2


# ─────────────────────────────────────────
# 学習パイプライン
# ─────────────────────────────────────────
def run(data_dir: str | None = None) -> None:
    """
    データ読み込みからモデル評価・SHAP分析まで一括実行する。

    Parameters
    ----------
    data_dir : str | None
        データCSVが置かれたディレクトリ。
        None の場合は .env の DATA_DIR を使用する（ローカル実行時）。
    """
    # data_dir が渡されなければ .env から取得
    if data_dir is None:
        data_dir = get_data_dir()

    print(f"[INFO] データディレクトリ: {data_dir}")

    # 1. DuckDB 初期化
    con = init_db(data_dir)

    # 2. データ読み込み
    load_salary(con, data_dir)
    load_bref_stats(con, data_dir)
    load_fangraphs_stats(con, data_dir)

    # 3. 前処理・結合
    df = build_model_df(con)

    # 4. 特徴量・ターゲット生成
    X, y = build_features(df)

    # 5. Train / Test 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"学習: {len(X_train)} 件  /  テスト: {len(X_test)} 件")

    # 6. モデル学習
    model_rf  = build_random_forest()
    model_lgb = build_lightgbm()
    model_xgb = build_xgboost()

    print("Training Random Forest ...")
    model_rf.fit(X_train, y_train)

    print("Training LightGBM ...")
    model_lgb.fit(X_train, y_train)

    print("Training XGBoost ...")
    model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 7. 予測
    preds_log = {
        "Random Forest": model_rf.predict(X_test),
        "LightGBM":      model_lgb.predict(X_test),
        "XGBoost":       model_xgb.predict(X_test),
    }

    # 8. 評価サマリー
    summary = evaluate_models({}, preds_log, y_test)
    _print_summary(summary)

    # 9. 特徴量重要度（Random Forest）
    plot_feature_importance(model_rf, X_train)

    # 10. 残差分析（Random Forest）
    residuals = build_residuals_df(y_test, preds_log["Random Forest"], df)
    plot_residuals(residuals)
    print_top_errors(residuals)

    # 11. SHAP分析（XGBoost）
    run_shap_analysis(model_xgb, X_test, residuals)

    # 12. DB を保存（Colab なら Drive へ同期）
    sync_db(con, data_dir)


# ─────────────────────────────────────────
# 可視化ヘルパー
# ─────────────────────────────────────────
def _print_summary(summary: pd.DataFrame) -> None:
    disp = summary.copy()
    disp["MAE (USD)"] = disp["MAE (USD)"].apply(lambda x: f"${x:,.0f}")
    disp["MAPE (%)"]  = disp["MAPE (%)"].apply(lambda x: f"{x:.2f}%")
    disp["R2 Score"]  = disp["R2 Score"].apply(lambda x: f"{x:.4f}")
    print("=" * 60)
    print("      MLB Salary Prediction: Final Evaluation")
    print("=" * 60)
    print(disp.to_string(index=False))
    print("-" * 60)


def plot_feature_importance(model, X_train: pd.DataFrame, top_n: int = 20) -> None:
    fi_df = pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_})
    top = fi_df.sort_values("Importance", ascending=False).head(top_n)
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=top, palette="viridis")
    plt.title(f"Top {top_n} Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.show()


def plot_residuals(residuals: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=residuals, x="Salary_Range", y="Residual", palette="vlag")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals by Salary Range (Actual − Predicted)", fontsize=15)
    plt.ylabel("Residual ($)")
    plt.xlabel("Actual Salary Range")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\n--- 年俸帯別・残差統計 ---")
    print(residuals.groupby("Salary_Range")["Residual"].agg(["mean", "std", "count"]))


def print_top_errors(residuals: pd.DataFrame, n: int = 10) -> None:
    top = residuals.sort_values("Absolute_Error", ascending=False).head(n)
    print(f"\n--- 予測誤差が大きい選手 TOP {n} ---")
    print(top[["Name_Clean", "Age", "Team", "Actual", "Predicted", "Residual"]])


def run_shap_analysis(model, X_test: pd.DataFrame, residuals: pd.DataFrame) -> None:
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_exp    = explainer(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="dot")

    top_errors = residuals.sort_values("Absolute_Error", ascending=False).head(10)
    print("\n--- 予測誤差が大きい選手 TOP 10 の SHAP 分析 ---")
    for rank, (orig_idx, row) in enumerate(top_errors.iterrows(), start=1):
        row_pos = X_test.index.get_loc(orig_idx)
        print(f"\nNo.{rank}: {row['Name_Clean']}")
        print(f"  Actual: ${row['Actual']:,.0f}  /  Predicted: ${row['Predicted']:,.0f}")
        shap.plots.waterfall(shap_exp[row_pos], show=False)
        plt.title(f"Rank {rank}: {row['Name_Clean']}", fontsize=15)
        plt.show()


# ─────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────
if __name__ == "__main__":
    run()  # .env の DATA_DIR を自動参照
