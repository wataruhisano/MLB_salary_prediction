# ⚾ MLB Salary Prediction

MLB（メジャーリーグベースボール）野手の統計データをもとに、機械学習で選手の適正年俸を予測するプロジェクトです。

## 概要

複数のデータソース（Baseball Reference・FanGraphs・Kaggle）から取得した打撃・守備・走塁指標と過去の受賞歴を統合し、選手のパフォーマンスが市場価値にどのように反映されるかを定量的に分析します。

## 背景・目的

MLBでは年俸の格差が非常に大きく、同じ成績の選手でも年齢・ポジション・受賞歴によって市場価値が大きく異なります。本プロジェクトでは以下を目的として分析を行いました。

- 選手の成績指標から年俸を定量的に予測するモデルの構築
- 年俸に影響を与える要因の特定と可視化
- 実績スコア（受賞歴）や年齢効果など、統計指標以外の要素の寄与度の検証

## リポジトリ構成

```
mlb_salary_prediction/
├── data/                     # データCSV置き場（.gitignore で除外）
│   └── .gitkeep
├── notebooks/
│   └── exploration.ipynb     # 試行錯誤・EDA・グラフ確認用ノートブック
├── src/
│   ├── preprocess.py         # データ読み込み・クリーニング・特徴量エンジニアリング
│   ├── model.py              # モデル定義・評価ユーティリティ
│   └── train.py              # 学習・評価・SHAP分析のエントリーポイント
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 使用技術

| カテゴリ | ライブラリ |
|---|---|
| 言語 | Python 3.12 |
| データ処理 | Pandas, NumPy, DuckDB |
| 機械学習 | LightGBM, XGBoost, scikit-learn |
| モデル解釈 | SHAP |
| 可視化 | Matplotlib, Seaborn |
| データ取得 | pybaseball, mlb-statsapi |

**開発環境**: Google Colab / VS Code

## データセット

著作権の都合上、データファイルはリポジトリに含めていません。以下のサイトから取得して `data/` フォルダに配置してください（対象年度: 2017〜2024、2020年除く）。

| # | データソース | 内容 | URL |
|---|---|---|---|
| 1 | Baseball Reference | Player Value Batting（基本打撃指標・受賞歴）| https://www.baseball-reference.com/ |
| 2 | FanGraphs | セイバーメトリクス指標（wOBA, wRC+, WAR 等）| https://www.fangraphs.com/leaders/major-league |
| 3 | Kaggle (Salary Data) | MLB選手の年度別年俸データ | https://www.kaggle.com/datasets/christophertreasure/mlb-player-salaries-2011-2024 |

取得後のファイル配置例：

```
data/
├── mlb_salary_data.csv
├── 2017_stats.csv
├── 2018_stats.csv
│   ...
├── fg_stats_2017.csv
├── fg_stats_2018.csv
│   ...
```

## 実装のポイント

### 効率的なデータパイプライン
DuckDB を採用し、複数の大規模 CSV ファイルを SQL ベースで高速に統合・集計しています。Pandas の DataFrame と DuckDB を組み合わせることで、結合・重複排除・ウィンドウ関数を一括処理しています。

### 特徴量エンジニアリング

| 特徴量 | 説明 |
|---|---|
| Age_Squared | 年齢の二乗項。加齢による非線形な能力低下を表現 |
| is_veteran | 28歳以上をベテランとして区別するバイナリフラグ |
| career_award_score | 過去5年の受賞歴をポイント化（MVP×10 / SS×3 / GG×2 / AS×1）|
| Pos_Group | BRef のポジションコードを Catcher / Infield / Outfield / DH に集約 |

### データクレンジング
- 正規表現による選手名末尾の記号除去（`Matt Wallner*` → `Matt Wallner`）
- `$5,850,000` 形式の文字列を数値に変換
- シーズン途中の移籍選手で生じる重複データを、打席数（PA）が多い行を優先する `ROW_NUMBER()` で除去
- アワード歴がない若手選手の欠損値を 0 補完

## 分析手法

1. **データ収集**: Baseball Reference・FanGraphs・Kaggle から 2017〜2024 年のデータを取得
2. **前処理**: DuckDB によるデータ結合・型変換・名寄せ
3. **特徴量設計**: ドメイン知識に基づく主要指標の選定と新規特徴量の生成
4. **多重共線性の診断**: VIF（分散膨張係数）および相関行列で特徴量間の冗長性を確認し、最終的に10特徴量に絞り込み
5. **学習・評価**: Random Forest・LightGBM・XGBoost の3モデルで比較し、MAE・MAPE・R² で評価
6. **重要度分析**: 特徴量インポータンスおよび SHAP によるモデルの解釈

## モデルの改善過程

| フェーズ | 施策 | 効果 |
|---|---|---|
| ベースライン | 成績指標のみで予測 | 高年俸選手の予測精度が低い |
| 特徴量追加 | 年齢の二乗項・ベテランフラグの追加 | 年齢カーブをより正確に表現 |
| 受賞歴の組み込み | career_award_score の導入 | 実績に基づく市場価値のブレを軽減 |
| 多重共線性の排除 | VIF 診断による特徴量の絞り込み | モデルの安定性が向上 |
| 目的変数の変換 | 年俸を log1p 変換してから学習 | 高年俸の外れ値による影響を緩和 |

## 結論と今後の展望

### 結論
打撃指標（wRC+）・総合貢献度（WAR）・出場機会（PA）が年俸予測に最も強く寄与することが確認されました。また、受賞歴スコアや年齢の二乗項も一定の説明力を持ち、純粋な成績指標だけでは捉えられないブランド価値や将来性への市場評価を部分的に反映できることが示されました。

### 今後の展望
- **守備指標の精緻化**: OAA（Outs Above Average）など新世代の守備指標を追加
- **サービスタイムの組み込み**: メジャー登録日数による年俸抑制効果（アービトレーション・FA資格）をモデルに反映
- **投手データへの拡張**: 現在は野手のみを対象としており、投手データへの応用が次のステップ
- **時系列構造の活用**: 前年比の成績変化率を特徴量に加え、トレンドを考慮した予測を実現

## セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/あなたのユーザー名/mlb_salary_prediction.git
cd mlb_salary_prediction

# 2. 仮想環境を作成・有効化
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. ライブラリをインストール
pip install -r requirements.txt

# 4. データを data/ フォルダに配置（上記「データセット」参照）
```

## 実行方法

### VS Code（ローカル）

**① `.env` を作成する**

```bash
cp .env.example .env
```

`.env` を開いて `DATA_DIR` を設定します（デフォルトは `./data`）。
DATA_DIR=./data
DB_PATH=./mlb_analytics.duckdb
**② 実行**

```bash
python src/train.py
```

### Google Colab

```python
!git clone https://github.com/あなたのユーザー名/mlb_salary_prediction.git
%cd mlb_salary_prediction
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/mlb_salary_prediction/src')

from train import run
run(data_dir="/content/drive/My Drive/個人開発/MLB_DATA_PREDICT/data/")
```

---

実行すると以下が順番に行われます。

1. DuckDB 初期化（既存 DB があれば再利用）
2. 年俸・BRef・FanGraphs データの読み込みと結合
3. 前処理・特徴量エンジニアリング
4. Train / Test 分割（8:2）
5. 3モデルの学習
6. 評価サマリー（MAE / MAPE / R²）の表示
7. 特徴量重要度の可視化
8. 残差分析・誤差が大きい選手の特定
9. SHAP によるモデル解釈
10. DuckDB を保存（Colab の場合は Drive へ同期）