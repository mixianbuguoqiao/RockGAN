import numpy as np
import math, sys, time
import pp

import math
import random

import matplotlib.pyplot as plt

INFINITE = 1000


import importlib
gan = importlib.import_module("3dgan_mit_biasfree")


result_generater = gan.ganGetOneResult("./ServerModel/biasfree_9610.cptk")

def RequestGANGen(lenX,lenY,lenZ):
    xMin, yMin, zMin, xMax, yMax, zMax = transfer(lenX,lenY,lenZ)
    # gan.SetBoundState(minx, miny, minz, maxx, maxy, maxz)
    gan.SetBoundState(xMin, yMin, zMin, xMax, yMax, zMax)
    result_list = next(result_generater)
    result_list = result_list.ravel()
    back = ""
    for b in result_list:
        if b == True:
            back = back + "1"
        else:
            back = back + "0"
    return back


def indicator(str,lenX,lenY,lenZ):


    result = str

    xMin, yMin, zMin, xMax, yMax, zMax = transfer(lenX,lenY,lenZ)



    npResult = np.zeros([64,64,64],dtype=bool)
    for x in range(64):
        for y in range(64):
                for z in range(64):
                    index = 64 * 64 * x + 64 * y + z
                    if(result[index] == '1'):
                      npResult[x][y][z] = True
                    if(result[index] == '0'):
                      npResult[x][y][z] = False

    all = 0  #所有的噪点总数
    num = 0  #包围盒外的噪点总数
    blacksize = 0 #包围盒外所有的点总数
    for x in range(64):
        for y in range(64):
            for z in range(64):
                if(npResult[x][y][z] == True):
                    all = all + 1
                if xMin<=x<= xMax and yMin<= y <=  yMax and zMin<=  z <=zMax:
                    pass
                else:
                    blacksize = blacksize + 1
                    if npResult[x][y][z] == True:
                      num = num + 1
    np.save("result.npy", npResult)
    return num,blacksize,all

def cal(n):
     if not isinstance(n, int):
        raise TypeError("argument passed to is_prime is not of 'int' type")
     list = []
     if n == 1:
         list.append((1,17))

     if n == 2 :
         list.append((17,33))

     if n == 3:
         list.append((33,49))

     if n == 4:
         list.append((49,65))

     listX=[]
     listY=[]
     setX = set([])
     setY = []

     load = 0

     for x  in range(list[0][0],list[0][1]):
        for y in range(1,65):
            for z in range(1,65):
               load = load + 1
               lenX = len(setX)

               listX.append( x * y * z )
               result = RequestGANGen(x,y,z)
               num=indicator(result,x,y,z)
               yAxis = num[0] / num[2]
               if num[0] == 0:
                 listY.append(INFINITE)
               else:
                 listY.append(-math.log(yAxis))

               setX.add( x * y * z )
               if len(setX)-lenX == 1 :
                   setY.append(-math.log(yAxis))
               print("*********",n,"：*******",load/(65536) *100,"*****************************")

     strX = "listX"+str(n)+".txt"
     fileX = open(strX,'w')
     fileX.write(str(listX))
     fileX.close()

     strY = "listY"+str(n)+".txt"
     fileY = open(strY,'w')
     fileY.write(str(listY))
     fileY.close()

     strFileX = "setX"+str(n)+".txt"
     fileSetX = open(strFileX,'w')
     fileSetX.write(str(setX))
     fileSetX.close()

     strFileXY = "setY"+str(n)+".txt"
     filelistXY = open(strFileXY,'w')
     filelistXY.write(str(setY))
     filelistXY.close()


     fig = plt.figure()
     ax1 = fig.add_subplot(111)
    #设置标题
     ax1.set_title('Scatter Plot')
    #设置X轴标签
     plt.xlabel('volume')
    #设置Y轴标签
     plt.ylabel('better')
    #画散点图
     ax1.scatter(listX,listY,c = 'r',marker = 'o')
    #设置图标
     plt.legend('x1')
    #显示所画的图
     plt.show()
def transfer(lenX,lenY,lenZ, centerX = 32, centerY = 32 , centerZ = 0):

    xMin = centerX - lenX/2
    xMax = centerX + lenX/2
    yMin = centerY - lenY/2
    yMax = centerY + lenY/2
    zMin = 0
    zMax = lenZ

    return xMin ,yMin,zMin,xMax,yMax,zMax

# result = RequestGANGen(x, y, z)
# num = indicator(result, x, y, z)
sum = 0
for i in range(50):
    num,blacksize,all =indicator(RequestGANGen(50,40,40),50,40,40)
    t = num/all *100
    sum = sum + t
    print("盒外的噪点总数:",num)

    # print("盒外的点总数:",blacksize)
    print("噪点总数:",all)
    print("盒外噪点所占的比例:",t)
    print("********************************")

print("the average of noise:", sum / 50)

# num,blacksize,all =indicator(RequestGANGen(40,15,10),40,15,10)
# t = num/all *100
#
# print("盒外的噪点总数:",num)
#
# # print("盒外的点总数:",blacksize)
# print("噪点总数:",all)
# print("盒外噪点所占的比例:",t)
# print("********************************")