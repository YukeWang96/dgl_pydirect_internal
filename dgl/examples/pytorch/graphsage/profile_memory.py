import os
import xlwt
import csv
import time
import multiprocessing as mp

def runapp(command,v):
    os.system(command)
    print(command)
    v.value=1

def query_memory(i,v):
    command='nvidia-smi --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv | tee gpu-log.csv'
    writefile=open(str(i+1)+'.mem','a')
    os.system(command)
    while 1:
        os.system(command)
        myfile=open('gpu-log.csv','r')
        #next(myfile)
        myline=myfile.readline()
        for i in range(4):
            #row=next(myfile)
            myline=myfile.readline()
            row=myline.split(',')
            writefile.write(row[7])
            #writefile.write('\n')
        time.sleep(0.1)
        if v.value==1:
            break
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
    gpuinst='0'
    ngpu=4
    embed=128
    gpuinst='0'
    for i in range(1,ngpu):
        gpuinst=gpuinst+','+str(i)
    for i in range(10):
        source=mtx_path_list[i]
        for j in range(ngpu):
            os.system('touch intermediate'+str(j)+'.out')
        command='python memory_train.py --gpu '+gpuinst+' --graph-device cpu --data-device uva --source '+source+' --nfeats '+str(embed)
        v = mp.Value('i',0)
        process1 = mp.Process(target=runapp, args=(command,v))
        process2 = mp.Process(target=query_memory, args=(i,v))
        process1.start()
        process2.start()
        process1.join()
        process2.join()

if __name__ == '__main__':
    main()
