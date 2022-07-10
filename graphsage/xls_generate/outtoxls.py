import os
import xlwt

def findAllfile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield base+'/'+f

def main():

    base = '../outfile'

    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('multi_gpu')
    itr=0
    for i in findAllfile(base):
        outfile=open(i,'r')
        for k in range(5):
            log=outfile.readline()
            path=log.split('/')
            graph_name=path[-1]
            for j in range(2):
                log=outfile.readline()
                log=log.split()
                worksheet.write(itr+1,0,graph_name)
                worksheet.write(itr+1,1,log[3])
                worksheet.write(itr+1,2,log[4])
                worksheet.write(itr+1,3,log[5])
                worksheet.write(itr+1,4,log[0])
                worksheet.write(itr+1,5,log[1])
                worksheet.write(itr+1,6,log[6])
                #worksheet.write(itr+1,7,log[6])
                itr=itr+1
        for k in range(5):
            log=outfile.readline()
            for j in range(4):
                log=outfile.readline()
                log=log.split()
                worksheet.write(itr+1,0,graph_name)
                worksheet.write(itr+1,1,log[3])
                worksheet.write(itr+1,2,log[4])
                worksheet.write(itr+1,3,log[5])
                worksheet.write(itr+1,4,log[0])
                worksheet.write(itr+1,5,log[1])
                worksheet.write(itr+1,6,log[6])
                #worksheet.write(itr+1,7,log[6])
                itr=itr+1
        outfile.close()
    workbook.save('multi_gpu.xls')
    

if __name__ == '__main__':
    main()
