[サッカー軌道予測コンペティション](https://sites.google.com/view/stp-challenge)におけるCamellia Dragonsのコードです.

ベストモデルをHuggingfaceにアップロードしています.
https://huggingface.co/ReonOhashi/RobocupTrajectoryPrediction2025

# ファイルの構成
**前処理**
1. `data_download.ip`: 2024年度のロボカップ2Dデータをすべてダウンロードし, ゴールデータを50フレーム抽出する 
2. `data_upload.ipynb`: 1のデータをhuggingfaceにアップロードをする
3. `data_preprocess.ipynb`: 2からデータを読み込んでデータの前処理を行い`data`ディレクトリに保存する

**訓練**
- `train_GRU.ipynb`: `data`ディレクトリの内容からモデルの訓練を行う

**予測**
- `submission.ipynb`: 提出用のcsvファイル`submission_data`から読み込みテストデータに対する予測を行う. 結果は`submission_data_out`に作成される. `train_GRU.ipynb`で作成されたモデル, もしくは先述のHuggingfaceのものを使用するようにパスを変更してください. 

**その他ライブラリ**
- `process_data.py` : `data_download.ipynb`にてデータの前処理を行うモジュール
- `visualize.py` : データの可視化を行うモジュール