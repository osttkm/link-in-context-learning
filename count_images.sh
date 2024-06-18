#!/bin/bash

# 対象ディレクトリを指定
directory="/dataset/yyama_dataset/VI_images/VI_full_train_context"

# カウントする画像ファイルの拡張子
extensions=("jpg" "jpeg" "png" "gif")

# ファイルカウント用変数
count=0

# 拡張子ごとにループ
for ext in "${extensions[@]}"; do
    # 指定ディレクトリにある特定の拡張子を持つファイルの数をカウント
    count=$(($count + $(find "$directory" -maxdepth 5 -type f -iname "*.$ext" | wc -l)))
done

# 結果を表示
echo "画像ファイルの数: $count"
