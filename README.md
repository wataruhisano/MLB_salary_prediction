# MLB選手年俸予測モデルの構築と分析

## 概要
本プロジェクトは、メジャーリーグベースボール（MLB）に所属する野手の統計データに基づき、機械学習を用いて選手の適正年俸を予測するモデルを構築したものです。
複数のデータソース（Baseball Reference, FanGraphs等）から取得した指標を統合し、選手のパフォーマンスが市場価値にどのように反映されるかを定量的に分析することを目的としています。

## 背景・目的

## 使用技術
### 言語・ライブラリ
* **言語**: Python 3.12
* **データ処理**: Pandas, Numpy, DuckDB
* **機械学習**: LightGBM, Scikit-learn
* **可視化**: Matplotlib, Seaborn
* **データ取得**: pybaseball, mlb-statsapi

### 開発環境
* Google Colab

## データセット
以下の3種類のデータを統合して使用しています。
1. **Baseball Reference Stats**: Baseball ReferenceのPlayer Value Batting　基本的な打撃指標およびタイトル受賞歴（MVP, SS等）https://www.baseball-reference.com/　
2. **FanGraphs Stats**: Fangraphsのセイバーメトリクス指標（wOBA, wRC+, WAR等）https://www.fangraphs.com/leaders/major-league
3. **Salary Data**: MLB選手の年度別年俸データ https://www.kaggle.com/datasets/christophertreasure/mlb-player-salaries-2011-2024

## 実装のポイント
* **効率的なデータパイプライン**: DuckDBを採用し、複数の巨大なCSVファイルをSQLベースで高速に統合・集計しています。
* **特徴量エンジニアリング**: 
  * 年齢による衰えを考慮した「年齢の二乗項目」の追加。
  * 過去5年間のアワード受賞歴のポイント化（実績スコアの算出）。
* **データクレンジング**: 正規表現を用いた選手名の名寄せや、欠損値に対する適切なインピュテーション（補完）処理、シーズン途中に移籍した選手のデータの重複の回避を実施。

## 分析手法
1. **データ収集**: Baseball Reference、FanGraphs、kaggleのサイトから2017年から2024年までのデータを取得。
2. **前処理**: DuckDBによるデータ集計、型変換、表記ゆれの修正。
3. **特徴量設計**: ドメイン知識に基づいた主要指標の選定と新規特徴量の生成。
4. **学習・評価**: LightGBMを用いたモデル構築と、MAE（平均絶対誤差）による評価。
5. **重要度分析**: 特徴量インポータンスを可視化し、予測に寄与する要因を特定。

## モデルの改善過程

## 結論と今後の展望

今後の展望として、守備指標のさらなる精緻化や、サービスタイム（メジャー登録日数）による年俸抑制効果などの契約面での制約をモデルに組み込むことで、予測精度の向上が期待できます。
