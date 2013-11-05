#!/bin/bash
echo "yes" | python twerpy/twerpy.py setup -d clx.db
python twerpy/twerpy.py search-top-users finance -d clx.db > top_users.txt
python twerpy/twerpy.py search-top-users tech -d clx.db >> top_users.txt
python twerpy/twerpy.py search-top-users health -d clx.db >> top_users.txt
python twerpy/twerpy.py search-top-users entertainment -d clx.db >> top_users.txt
python twerpy/twerpy.py search-top-users sport -d clx.db >> top_users.txt
python twerpy/twerpy.py search-top-users politics -d clx.db >> top_users.txt

python twerpy/twerpy.py search-user-tweets top_users.txt -d clx.db
python twerpy/twerpy.py dump-tweets -o "tweets.csv" -d clx.db