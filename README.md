#  MLB Salary Prediction

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
├── data/                     # データCSV置き場
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

**開発環境**: Google Colab / VS Code

## データセット

利用規約の都合上、データファイルはリポジトリに含めていません。以下のサイトから取得して `data/` フォルダに配置してください（対象年度: 2017〜2024、コロナ禍による影響で他の年度より試合数が少ないため、2020年は除く）。

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
| is_veteran | 31歳以上をベテランとして区別するバイナリフラグ |
| career_award_score | 過去5年の受賞歴をポイント化（MVP×10 / SS×3 / GG×2 / AS×1）|
| Pos_Group | BRef のポジションコードを Catcher / First Base / Second Base / Third Base / Shortstop / Outfield / DH に分類 |

### データクレンジング
- 正規表現による選手名末尾の記号除去（`Matt Wallner*` → `Matt Wallner`）
- `$5,850,000` 形式の文字列を数値に変換
- シーズン途中の移籍選手で生じる重複データを、打席数（PA）が多い行を優先する `ROW_NUMBER()` で除去
- 年度別の最低年俸（2017: $535,000 〜 2024: $740,000）を基準に最低年俸未満のレコードを除去
- アワード歴がない若手選手の欠損値を 0 補完

## 分析手法

1. **データ収集**: Baseball Reference・FanGraphs・Kaggle から 2017〜2024 年のデータを取得
2. **前処理**: DuckDB によるデータ結合・型変換・名寄せ
3. **特徴量設計**: ドメイン知識に基づく主要指標の選定と新規特徴量の生成
4. **多重共線性の診断**: VIF（分散膨張係数）および相関行列で特徴量間の冗長性を確認し、最終的に10特徴量に絞り込み
5. **学習・評価**: 年度別最低年俸でインフレ調整した実質年俸を目的変数とし、TimeSeriesSplit（5分割交差検証）で Random Forest・LightGBM・XGBoost の3モデルを比較。MAE・MAPE・R² の平均と標準偏差で評価
6. **重要度分析**: 特徴量インポータンスおよび SHAP によるモデルの解釈

## モデルの改善過程

| フェーズ | 施策 | 効果 |
|---|---|---|
| ベースライン | 成績指標や守備位置で予測 | 高年俸選手の予測精度が低い |
| 受賞歴の組み込み | career_award_score の導入 | 実績に基づく市場価値のブレを軽減 |

**最終モデルの交差検証結果（TimeSeriesSplit 5-Fold）**

| Model | MAE mean | MAE std | MAPE mean | R2 mean |
|---|---|---|---|---|
| Random Forest | $4,893,407 | ±$734,828 | 102.46% | 0.5345 |
| LightGBM | $5,260,755 | ±$904,530 | 117.73% | 0.4744 |
| XGBoost | $5,395,648 | ±$751,697 | 121.56% | 0.4182 |

## 結論と今後の展望

### 結論

3モデルの比較では、Random Forest が最も高い予測精度を示しました（MAE: $4,893,407 / R²: 0.5345）。成績指標に加えて受賞歴スコアを取り入れることで、ベースラインからR²を約0.13改善しました。

一方で MAPE が100%を超えている点から、特に低〜中年俸帯（$2M 未満）の選手については予測誤差が大きい傾向があります。これはミニマム契約に近い若手選手の年俸が成績指標だけでは説明しにくい契約上の事情（サービスタイム・アービトレーション）に左右されることが主な原因と考えられます。

### 今後の展望
- **守備指標の精緻化**: OAA（Outs Above Average）など新世代の守備指標を追加
- **サービスタイムの組み込み**: メジャー登録日数による年俸抑制効果（アービトレーション・FA資格）をモデルに反映
- **投手データへの拡張**: 現在は野手のみを対象としており、投手データへの応用が次のステップ
- **時系列構造の活用**: 前年比の成績変化率を特徴量に加え、トレンドを考慮した予測を実現

## セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/wataruhisano/mlb_salary_prediction.git
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

```
DATA_DIR=./data
DB_PATH=./mlb_analytics.duckdb
```

**② 実行**

```bash
python src/train.py
```

### Google Colab

```python
!git clone https://github.com/wataruhisano/mlb_salary_prediction.git
%cd mlb_salary_prediction
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/mlb_salary_prediction/src')

from train import run
DATA_DIR = "/content/drive/My Drive/ここにあなたのデータフォルダのパスを入力/"
run(data_dir=DATA_DIR)
```

---

実行すると以下が順番に行われます。

1. DuckDB 初期化（既存 DB があれば再利用）
2. 年俸・BRef・FanGraphs データの読み込みと結合
3. 前処理・特徴量エンジニアリング
4. インフレ調整（年度別最低年俸を基準に実質年俸へ変換）
5. TimeSeriesSplit（5分割交差検証）による3モデルの学習・評価
6. 評価サマリー（MAE / MAPE / R² の平均・標準偏差）の表示
7. 特徴量重要度の可視化
8. 残差分析・誤差が大きい選手の特定
9. SHAP によるモデル解釈
10. DuckDB を保存（Colab の場合は Drive へ同期）
