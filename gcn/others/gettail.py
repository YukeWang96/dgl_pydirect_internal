import os
import csv

def main():
    for i in range(10):
        path='memout/'+str(i+1)+'.mem'
        command='tail -n 100 '+path+' >memout/'+str(i+1)+'.newmem'
        os.system(command)

if __name__ == '__main__':
    main()
