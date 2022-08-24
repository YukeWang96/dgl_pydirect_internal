import os
import csv

def findAllfile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield base+'/'+f

def main():

    mtx_path_list=[
    'wiki-talk-temporal',
    'soc-LiveJournal1',
    'com-LiveJournal',
    'cit-Patents',
    'sx-stackoverflow',
    'wiki-Talk',
    'roadNet-CA',
    'wiki-topcats',
    'as-Skitter',
    'roadNet-TX',
    'papers100m',
    'mag240m',
    'twitter7'
    ]
    log_path = 'pytorch_direct_result.csv'
    file = open(log_path, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['graph_name','num_node','num_edge','num_gpu','embedding_size','time(ms)'])
    embedding_list=[16,32,64,128,256]
    #2gpu
    for i in range(10):
        path='2gpu/'+str(i+1)+'.out'
        outfile=open(path,'r')
        for j in range(5):
            log=outfile.readline()
            max_time=0
            for k in range(2):
                log=outfile.readline()
                log=log.split()
                acc=float(log[6])+float(log[7])
                if acc>max_time:
                    max_time=acc
            csv_writer.writerow([mtx_path_list[i],log[3],log[4],log[0],log[2],str(max_time)])

    #4gpu
    for i in range(10):
        path='4gpu/'+str(i+1)+'.out'
        outfile=open(path,'r')
        for j in range(5):
            log=outfile.readline()
            max_time=0
            for k in range(4):
                log=outfile.readline()
                log=log.split()
                acc=float(log[6])+float(log[7])
                if acc>max_time:
                    max_time=acc
            csv_writer.writerow([mtx_path_list[i],log[3],log[4],log[0],log[2],str(max_time)])

    #8gpu
    for i in range(10):
        path='8gpu/'+str(i+1)+'.out'
        outfile=open(path,'r')
        for j in range(5):
            log=outfile.readline()
            max_time=0
            for k in range(8):
                log=outfile.readline()
                log=log.split()
                acc=float(log[6])+float(log[7])
                if acc>max_time:
                    max_time=acc
            csv_writer.writerow([mtx_path_list[i],log[3],log[4],log[0],log[2],str(max_time)])

    #large
    for i in range(10,13):
        path='large/'+str(i+1)+'.out'
        outfile=open(path,'r')
        for j in range(5):
            log=outfile.readline()
            max_time=0
            oom=0
            for k in range(8):
                log=outfile.readline()
                log=log.split()
                if len(log)<8:
                    oom=1
                if oom!=1:
                    acc=float(log[6])+float(log[7])
                    if acc>max_time:
                        max_time=acc
            if oom!=1:
                csv_writer.writerow([mtx_path_list[i],log[3],log[4],log[0],log[2],str(max_time)])
            else:
                csv_writer.writerow([mtx_path_list[i],'oom','oom','oom','oom','oom'])


    file.close()

if __name__ == '__main__':
    main()
