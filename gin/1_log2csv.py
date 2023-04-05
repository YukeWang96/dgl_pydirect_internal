#!/usr/bin/env python3
import os
import sys 

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_log2csv.py result.log")

fp = open(sys.argv[1], "r")

fetch_li = []
aggre_li = []
data_li = []
for line in fp:
    if "Fetch (ms):" in line:
        time = line.split("Fetch (ms):")[1].rstrip("\n").lstrip()
        # print(time)
        fetch_li.append(float(time))
    if "Aggre (ms):" in line:
        time = line.split("Aggre (ms):")[1].rstrip("\n").lstrip()
        # print(time)
        aggre_li.append(float(time))
    if "dataset:" in line:
        data = line.split("dataset:")[1].rstrip("\n").lstrip()
        print(data)
        data_li.append(data)
fp.close()

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
# for fetch, aggre in zip(fetch_li, aggre_li):
#     fout.write("{},{}\n".format(fetch, aggre))
# fout.write("dataset, time(ms)\n")
for data, fetch, aggre in zip(data_li, fetch_li, aggre_li):
    fout.write("{},{:.3f}\n".format(data, fetch+aggre))
fout.close()
# os.system("rm *.out")
