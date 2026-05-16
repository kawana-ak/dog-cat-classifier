# dog-cat-classifier

## 概要
学習用に作成したTensorFlow / Keras を用いて作成した犬猫分類モデルです。  
EfficientNetB3 をベースに転移学習とファインチューニングを実施しました。

## 使用技術
- Python
- TensorFlow / Keras
- EfficientNetB3
- Kaggle

## 工夫した点
- 過学習対策として以下を実装
  - Dropout
  - Data Augmentation
  - Label Smoothing
  - EarlyStopping
- VGG16では精度が低かったため EfficientNetB3 に変更
- 転移学習実装時精度が95%だった為、ファインチューニングを実施しaccuracy99%を目標とした
- 学習率スケジューラを導入

## 学習結果
- accuracy: 0.9989
- loss: 0.209

## 備考
- 実装時に生成AIを活用
- Kaggle Notebook 上で学習を実施
