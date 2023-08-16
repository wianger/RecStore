ps aux |grep python |grep -v grep | awk '{print $2}' |xargs kill -9

while [ $? -eq 0 ]; do
    sleep 1s
    echo "waiting for killing process"
    ps aux |grep python |grep -v grep | awk '{print $2}' |xargs kill -9
done




pytest --count=10 -s  test_emb.py

