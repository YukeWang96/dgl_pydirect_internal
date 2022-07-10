'''import os
import xlwt

def main():

    base = 'graphdata/snap/'
    
    itr=0
    mtx_path_list=[
    #'graphdata/snap/email-Eu-core/email-Eu-core.mtx',
    #'graphdata/snap/wiki-talk-temporal/wiki-talk-temporal.mtx',
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
    for i in range(10):
        source=mtx_path_list[i]
        os.system('touch intermediate.out')
        command='python train_sampling_multi_gpu.py --graph-device uva --data-device uva --source '+source
        os.system(command)
        print(command)

        myfile=open('intermediate0.out','r')
        writefile=open(str(i+1)+'.out','w')
        writefile.write(source)
        writefile.write('\n')
        myline=myfile.readline()
        myline=myfile.readline()
        print('content')
        print(myline)
        writefile.write(myline)
        myfile.close()
        writefile.close()
        os.system('rm intermediate0.out')
        break

if __name__ == '__main__':
    main()
'''
import os
import xlwt

def main():

    base = 'graphdata/snap/'

    itr=0
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
    #ngpu=1
    gpuinst='0'
    gpu_list=[2,4]
    embedding_size=[16,32,64,128,256]
    for n in range(2):
        ngpu=gpu_list[n]
        gpuinst='0'
        for i in range(1,ngpu):
            gpuinst=gpuinst+','+str(i)
        for i in range(10):
            for k in range(5):
                embed=embedding_size[k]
                source=mtx_path_list[i]
                for j in range(ngpu):
                    os.system('touch intermediate'+str(j)+'.out')
                command='python train_sampling_multi_gpu.py --gpu '+gpuinst+' --graph-device uva --data-device uva --source '+source+' --nfeats '+str(embed)
                os.system(command)
                print(command)

                writefile=open(str(i+1)+'.out','a')
                writefile.write(source)
                writefile.write('\n')
                for j in range(ngpu):
                    myfile=open('intermediate'+str(j)+'.out','r')
                    #writefile=open(str(i+1)+'.out','w')
                    writefile.write(str(ngpu)+' '+str(j)+' '+str(embed)+' ')
                    myline=myfile.readline()
                    myline=myfile.readline()
                    print('content')
                    print(myline)
                    writefile.write(myline)
                    writefile.write('\n')
                    myfile.close()
                writefile.close()
                for j in range(ngpu):
                    os.system('rm intermediate'+str(j)+'.out')
                #os.system('rm intermediate0.out')
                #break

if __name__ == '__main__':
    main()
