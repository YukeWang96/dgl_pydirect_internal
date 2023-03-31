#!/usr/bin/env python3
import re
import sys 

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_log2csv.py result.log")

fp = open(sys.argv[1], "r")

fetch_li = []
aggre_li = []
for line in fp:
    if "Fetch (ms):" in line:
        time = line.split("Fetch (ms):")[1].rstrip("\n").lstrip()
        print(time)
        fetch_li.append(time)
    if "Aggre (ms):" in line:
        time = line.split("Aggre (ms):")[1].rstrip("\n").lstrip()
        print(time)
        aggre_li.append(time)
fp.close()

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
for fetch, aggre in zip(fetch_li, aggre_li):
    fout.write("{},{}\n".format(fetch, aggre))

fout.close()