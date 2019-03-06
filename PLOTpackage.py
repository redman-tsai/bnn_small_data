import matplotlib.pyplot as plt
import re

#----------------readData

def extractFile(file):
    content=[]
    with open(file,'r') as F:
        next(F)
        content=F.readlines()
        print(content)
    retData=[]
    for item in content:
        pattern=r'[\d.]+'
        a=re.findall(pattern,item)
        retData.append(a[-1])
    return retData[10::11]


BNN=extractFile('2017-07-26_17:17:08_OutputBNN_Deep.txt')
ANN=extractFile('2017-07-26_17:17:13_OutputANN_Deep.txt')
print(BNN)
print(ANN)

import numpy as np
def plotResult(ANN,BNN):
    plt.figure(1)
    plt.plot(ANN,'blue')#2Vi,3Vj,4Vk
    plt.plot(BNN,'red')
    plt.show()

plotResult(ANN,BNN)


