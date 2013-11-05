#!/bin/bash
echo "yes" | python twerpy/twerpy.py setup -d clx_sug.db
python twerpy/twerpy.py search-suggested-users -d clx_sug.db > users.txt
python twerpy/twerpy.py search-user-tweets users.txt -d clx_sug.db
python twerpy/twerpy.py dump-tweets -o "tweets_sug.csv" -d clx_sug.db