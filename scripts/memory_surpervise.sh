#!/bin/bash

# 監視対象のディレクトリを設定
WATCH_DIR="/data01/"

# inotifywaitでディレクトリの変更を監視
inotifywait -m -e open --format '%w%f' "$WATCH_DIR" | while read FILE
do
  # ファイルが開かれたらstraceで読み取りイベントを監視
  if [ -f "$FILE" ]; then
    echo "Monitoring read events for $FILE"
    strace -e trace=read -p $(pgrep -f "cat $FILE") &
  fi
done
