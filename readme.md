# 
サッカー軌道予測コンペティションにおけるCamelliDragonsのコードです.

# 
ベストモデルをデータをhuggingfaceに公開しています. 

# 注意事項
- wandb及びhuggingfaceのログインが求められるかもしれません. 

# ファイルの構成
**前処理**
- `data_download.ipynb`: 全データのダウンロード & ゴールデータを50フレーム抽出する 
- `data_upload.ipynb`: データをhuggingfaceにアップロードを行う
- `data_preprocess.ipynb`: データの前処理を行い`data`ディレクトリに保存する. 

**訓練**
- `train.ipynb`: モデルの訓練を行う。

**予測**
- `submission.ipynb`: テストデータに対する予測を行い、提出用のcsvファイル`submission_data`から読み込み, `submission_data_out`に作成する。モデルのパスを適切に変更してください。

**その他ライブラリ**
- `process_data.py` : `data_download.ipynb`にて, データの前処理を行うモジュール
- `visualize.py` : データの可視化を行うモジュール