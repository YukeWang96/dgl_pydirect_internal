import os
import csv
import re

def main():
    mtx_path_list=[
    #'graphdata/snap/email-Eu-core/email-Eu-core.mtx',
    'graphdata/snap/wiki-talk-temporal/wiki-talk-temporal.mtx',
    'graphdata/snap/soc-LiveJournal1/soc-LiveJournal1.mtx',
    'graphdata/snap/com-LiveJournal/com-LiveJournal.mtx',
    'graphdata/snap/cit-Patents/cit-Patents.mtx',
    'graphdata/snap/sx-stackoverflow/sx-stackoverflow.mtx',
    'graphdata/snap/wiki-Talk/wiki-Talk.mtx',
    'graphdata/snap/roadNet-CA/roadNet-CA.mtx',
    'graphdata/snap/wiki-topcats/wiki-topcats.mtx',
    'graphdata/snap/as-Skitter/as-Skitter.mtx',
    'graphdata/snap/roadNet-TX/roadNet-TX.mtx'
    ]
    log_path = 'memory.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['graph_name','num_gpu','memory(MB)','memory_util(%)'])
    for i in range(10):
        path='memout/'+str(i+1)+'.newmem'
        outfile=open(path,'r')
        gpu0_max=0
        gpu1_max=0
        gpu2_max=0
        gpu3_max=0
        for j in range(25):
            log=outfile.readline()
            gpu0=(re.findall(r"\d+\.?\d*",log))[0]
            log=outfile.readline()
            gpu1=(re.findall(r"\d+\.?\d*",log))[0]
            log=outfile.readline()
            gpu2=(re.findall(r"\d+\.?\d*",log))[0]
            log=outfile.readline()
            gpu3=(re.findall(r"\d+\.?\d*",log))[0]
            if int(gpu0)>gpu0_max:
                gpu0_max=int(gpu0)
            if int(gpu1)>gpu1_max:
                gpu1_max=int(gpu1)
            if int(gpu2)>gpu2_max:
                gpu2_max=int(gpu2)
            if int(gpu3)>gpu3_max:
                gpu3_max=int(gpu3)
        avg=(int(gpu0_max)+int(gpu1_max)+int(gpu2_max)+int(gpu3_max))/4
        avg_util=avg/40960*100
        n_gpu=4
        graph_name=(mtx_path_list[i].split('/'))[-1]
        csv_writer.writerow([graph_name,str(n_gpu),str(avg),str(avg_util)])
        outfile.close()
    file.close()

if __name__ == '__main__':
    main()
