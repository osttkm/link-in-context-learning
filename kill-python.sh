#このままだと拡張機能のpylanceとかも殺すので要改善
echo python task is killing...
echo $(ps -x | grep python | awk '{ print $1 }')
kill $(ps -x | grep python | awk '{ print $1 }')
# echo finish killing