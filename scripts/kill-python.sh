echo "Python tasks are being killed..."
ps -x | grep '.py' | awk '{ print $1 }'
kill $(ps -x | grep '.py' | awk '{ print $1 }')
echo "Finished killing Python tasks."