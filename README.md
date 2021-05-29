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
