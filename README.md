# Commonsense inference (Multiple-choice QA)
自然言語処理で使われる学習モデルを勉強するために書いたもの

常識推論のタスク（４択問題）を解くためのモデルを実装した
- MLPベースのモデル
- BERTベースのモデル

## Files
```
.
|- README.md
|- poetry.lock
|- pyproject.toml
|- data: データ置き場 gitignore に登録済み
|- src
   |- dataset.py
   |- model.py
   |- train_mlp.py
   |- test_mlp.py
   |- train_bert.py
   |- test_bert.py
```

## Requirements
Python 3.8 以上

## Installation
[Poetry](https://python-poetry.org/) での環境構築を推奨

```
$ git clone git@github.com:rik-tak/commonsence_inference.git
$ cd commonsence_inference
$ poetry install
```
