# -*- coding: utf-8 -*-
"""
Created on %(date)s
@email:guofang575856@163.com
@author: %('郭放')s
Strive for the goal
"""

import numpy as np
import random, math, copy
import matplotlib.pyplot as plt
import torch as t
import time
import pandas as pd

fname = "landing data.xlsx"
data = pd.read_excel(fname)
df = data.iloc[:, 4]

weight = np.array(df)
weight = np.around(weight, decimals=2)
weight = list(weight)
# print(weight)
height = len(weight)

one = np.ones(height)
t_glide = one * 111
t_glide = list(t_glide)
t_clear = one * 80
t_clear = list(t_clear)
t_sep = one * 94
t_sep = list(t_sep)
t_transfer = one * 236
t_transfer = list(t_transfer)


def seq_chrom(chrom):
    land_seq = np.argsort(chrom)
    return land_seq

def finishtime_chrom(seq):             #着舰完成时间
    dT = 1

    Vcarrier = 30  # 航母速度
    Ls = 127  # 舰载机相对航母的速度
    c = []
    T_leave_low = list(np.zeros(choromosome_length))
    T_transfer = list(np.zeros(choromosome_length))
    T_former_point = list(np.zeros(choromosome_length))
    T_landing = list(np.zeros(choromosome_length))
    t_level = list(np.zeros(choromosome_length))
    sum_level = 0.
    for j in range(choromosome_length):
        if j == 0:
            T_leave_low[seq[j]] = 0.
            T_transfer[seq[j]] = t_transfer[seq[j]]
            T_landing[seq[j]] = T_transfer[seq[j]] + t_glide[seq[j]]
        else:
            X = True
            sum_level = np.sum(t_level)
            T_leave_low[seq[j]] = T_leave_low[seq[j-1]] + t_sep[seq[j]] - 1
            while X:
                T_leave_low[seq[j]] += dT
                T_former_point[seq[j]] = T_leave_low[seq[j]] + t_transfer[seq[j]] + sum_level
                if T_former_point[seq[j]] >= (T_transfer[seq[j-1]]+t_sep[seq[j]]):
                    t_level[seq[j]] = (T_former_point[seq[j]] + t_glide[seq[j]] - T_landing[seq[j-1]]) * Vcarrier / Ls /(1 - Vcarrier / Ls)
                    T_transfer[seq[j]] = T_former_point[seq[j]] + t_level[seq[j]]
                    T_landing[seq[j]] = T_transfer[seq[j]] + t_glide[seq[j]]
                    if T_landing[seq[j]] >= (T_landing[seq[j-1]]+t_clear[seq[j]]):
                        X = False
    c = T_landing
    return c

def function_chrom(chrom):
    seq = seq_chrom(chrom)
    finishtime = finishtime_chrom(seq)
    # weight = [0.21, 0.43, 0.31, 0.54, 0.62, 0.15, 0.59, 0.31, 0.46, 0.18, 0.30]
    funct = 0
    for j in range(choromosome_length):
        funct += finishtime[j] * weight[seq[j]]
    return 1 / (1 + funct)

def generate(bound):
    chrom = [random.uniform(a,b) for a,b in zip(bound[0,:],bound[1,:])]
    return chrom

def calculateFitness(chrom):
    score = function_chrom(chrom)
    return score          #计算当前成绩

class ArtificialBeeSwarm:
    def __init__(self, foodCount, onlookerCount, bound, maxIterCount, maxInvalidCount):
        self.foodCount = foodCount                  #蜜源个数，等同于雇佣蜂数目
        self.onlookerCount = onlookerCount          #观察蜂个数
        self.bound = bound                          #各参数上下界
        self.maxIterCount = maxIterCount            #迭代次数
        self.maxInvalidCount = maxInvalidCount      #最大无效次数
        self.foodList = [generate(self.bound) for k in range(self.foodCount)]   #初始化各蜜源
        self.foodScore = [calculateFitness(d) for d in self.foodList]                             # 初始蜜源成绩，作为各蜜源历史最佳成绩
        self.bestFood = self.foodList[np.argmax(self.foodScore)]                      # 全局最佳蜜源
        # self.elite = self.foodList[np.argsort(self.foodScore)[-3:]]# 3个精英个体

    def updateFood(self, i):                                                  #更新第i个蜜源
        k = random.randint(0, self.bound.shape[1] - 1)                         #随机选择调整参数的维度
        j = random.choice([d for d in range(self.foodCount) if d !=i])   #随机选择另一蜜源作参考,j是其索引号
        vi = self.foodList[i]
        vi[k] = vi[k] + random.uniform(-1.0, 1.0) * (vi[k] - self.foodList[j][k])
        # + random.uniform(0, 1.5) * (self.bestFood[k] -vi[k])  # 调整参数
        vi[k] = np.clip(vi[k], self.bound[0, k], self.bound[1, k])               #参数不能越界
        # self.crossover(i)
        vi_score = calculateFitness(vi)
        if vi_score > calculateFitness(self.foodList[i]):           #如果成绩比当前蜜源好
            self.foodList[i] = vi
            if vi_score > self.foodScore[i]:            #如果成绩比历史成绩好（如重新初始化，当前成绩可能低于历史成绩）
                self.foodScore[i] = vi_score
                if vi_score > calculateFitness(self.bestFood):      #如果成绩全局最优
                    self.bestFood = vi
            invalidCount[i] = 0
        else:
            invalidCount[i] += 1

    def crossover(self, i):
        cross = random.randint(0, self.foodCount - 1)
        vi = self.foodList[i]
        if is_crossover > 0:
            if (random.random() < pc):  # 引入交叉
                cpoint = random.randint(0, self.bound.shape[1] - 1)
                temporary1 = []
                temporary2 = []
                temporary1.extend(vi[0:cpoint])
                temporary1.extend(self.foodList[cross][cpoint:len(vi)])
                temporary2.extend(self.foodList[cross][0:cpoint])
                temporary2.extend(vi[cpoint:len(vi)])
                vi = temporary1
                # self.foodList[i] = temporary2
        vi_score = calculateFitness(vi)
        if vi_score > calculateFitness(self.foodList[i]):           #如果成绩比当前蜜源好
            self.foodList[i] = vi
            if vi_score > self.foodScore[i]:            #如果成绩比历史成绩好（如重新初始化，当前成绩可能低于历史成绩）
                self.foodScore[i] = vi_score
                if vi_score > calculateFitness(self.bestFood):      #如果成绩全局最优
                    self.bestFood = vi
            invalidCount[i] = 0
        else:
            invalidCount[i] += 1

    def sj_twopoint_jh(self, i): # 随机两点交换
        # cross = random.randint(0, self.foodCount - 1)
        vi = self.foodList[i]
        point1 = random.randint(0, self.bound.shape[1] - 1)
        point2 = random.randint(0, self.bound.shape[1] - 1)
        if point1 != point2:
            x = vi[point1]
            y = vi[point2]
            vi[point2] = x
            vi[point1] = y
        vi_score = calculateFitness(vi)
        if vi_score > calculateFitness(self.foodList[i]):           #如果成绩比当前蜜源好
            self.foodList[i] = vi
            if vi_score > self.foodScore[i]:            #如果成绩比历史成绩好（如重新初始化，当前成绩可能低于历史成绩）
                self.foodScore[i] = vi_score
                if vi_score > calculateFitness(self.bestFood):      #如果成绩全局最优
                    self.bestFood = vi
            invalidCount[i] = 0
        else:
            invalidCount[i] += 1

    def sj_jubu_jh(self, i): # 随机局部交换,区连续三点
        vi = self.foodList[i]
        point1 = random.randint(0, self.bound.shape[1] - 1)
        point2 = random.randint(0, self.bound.shape[1] - 1)
        if point1 != point2:
            if point1 == 0:
                if point2 == (self.bound.shape[1] - 1):
                    x = vi[point1]
                    y = vi[point1 + 1]
                    z = vi[point1 + 2]
                    vi[point1] = vi[point2 - 2]
                    vi[point1 + 1] = vi[point2 - 1]
                    vi[point1 + 2] = vi[point2]
                    vi[point2 - 2] = x
                    vi[point2 - 1] = y
                    vi[point2] = z
                else:
                    x = vi[point1]
                    y = vi[point1 + 1]
                    z = vi[point1 + 2]
                    vi[point1] = vi[point2 - 1]
                    vi[point1 + 1] = vi[point2]
                    vi[point1 + 2] = vi[point2 + 1]
                    vi[point2 - 1] = x
                    vi[point2] = y
                    vi[point2 + 1] = z
            elif point2 == 0:
                if point1 == (self.bound.shape[1] - 1):
                    x = vi[point2]
                    y = vi[point2 + 1]
                    z = vi[point2 + 2]
                    vi[point2] = vi[point1 - 2]
                    vi[point2 + 1] = vi[point1 - 1]
                    vi[point2 + 2] = vi[point1]
                    vi[point1 - 2] = x
                    vi[point1 - 1] = y
                    vi[point1] = z
                else:
                    x = vi[point2]
                    y = vi[point2 + 1]
                    z = vi[point2 + 2]
                    vi[point2] = vi[point1 - 1]
                    vi[point2 + 1] = vi[point1]
                    vi[point2 + 2] = vi[point1 + 1]
                    vi[point1 - 1] = x
                    vi[point1] = y
                    vi[point1 + 1] = z
            elif point1 == (self.bound.shape[1] - 1) and point2 != 0:
                x = vi[point2 - 1]
                y = vi[point2]
                z = vi[point2 + 1]
                vi[point2 - 1] = vi[point1 - 2]
                vi[point2] = vi[point1 - 1]
                vi[point2 + 1] = vi[point1]
                vi[point1 - 2] = x
                vi[point1 - 1] = y
                vi[point1] = z
            elif point2 == (self.bound.shape[1] - 1) and point1 != 0:
                x = vi[point1 - 1]
                y = vi[point1]
                z = vi[point1 + 1]
                vi[point1 - 1] = vi[point2 - 2]
                vi[point1] = vi[point2 - 1]
                vi[point1 + 1] = vi[point2]
                vi[point2 - 2] = x
                vi[point2 - 1] = y
                vi[point2] = z
            else:
                x = vi[point1 - 1]
                y = vi[point1]
                z = vi[point1 + 1]
                vi[point1 - 1] = vi[point2 - 1]
                vi[point1] = vi[point2]
                vi[point1 + 1] = vi[point2 + 1]
                vi[point2 - 1] = x
                vi[point2] = y
                vi[point2 + 1] = z
        else:
            if point2 != 0 and point2 != (self.bound.shape[1] - 1):
                x = vi[point2 - 1]
                vi[point2 + 1] = vi[point2 - 1]
                vi[point2 - 1] = x
            elif point2 == 0:
                x = vi[point2]
                vi[point2] = vi[point2 + 2]
                vi[point2 + 2] = x
            elif point2 == (self.bound.shape[1] - 1):
                x = vi[point2]
                vi[point2] = vi[point2 - 2]
                vi[point2 - 2] = x
        vi_score = calculateFitness(vi)
        if vi_score > calculateFitness(self.foodList[i]):           #如果成绩比当前蜜源好
            self.foodList[i] = vi
            if vi_score > self.foodScore[i]:            #如果成绩比历史成绩好（如重新初始化，当前成绩可能低于历史成绩）
                self.foodScore[i] = vi_score
                if vi_score > calculateFitness(self.bestFood):      #如果成绩全局最优
                    self.bestFood = vi
            invalidCount[i] = 0
        else:
            invalidCount[i] += 1

    def elitepolicy(self): # 保留三个精英个体
        suoyin = np.argsort(self.foodScore)  # 从小到大排序的索引
        elite_ = suoyin[-2:] # 返回精英给提对应索引
        poor = suoyin[:2] # 返回最差个体对应索引
        for j in range(2):
            self.foodList[poor[j]] = self.foodList[elite_[j]]

    def employedBeePhase(self):
        for i in range(self.foodCount):              #各蜜源依次更新
            self.updateFood(i)
            self.crossover(i)

    def onlookerBeePhase(self):
        foodScore = [calculateFitness(d) for d in self.foodList]
        foodScore = np.array(foodScore)
#        maxScore = np.max(foodScore)
        sumscore = np.sum(foodScore)
        accuFitness = [(d/sumscore, k) for k,d in enumerate(foodScore)]
#        accuFitness = [(0.9*d/maxScore+0.1, k) for k,d in enumerate(foodScore)]        #得到各蜜源的 相对分数和索引号
        for i in range(self.onlookerCount):
            # i = random.randint(0, self.foodCount - 1)
            if accuFitness[i][0] > random.random():
                self.updateFood(i)
                self.sj_twopoint_jh(i)
                self.sj_jubu_jh(i)

    def scoutBeePhase(self):
        for i in range(self.foodCount):
            if invalidCount[i] > self.maxInvalidCount:                    #如果该蜜源没有更新的次数超过指定门限，则重新初始化
                self.foodList[i] = generate(self.bound)
                self.foodScore[i] = max(self.foodScore[i], calculateFitness(self.foodList[i]))  # 历史最优成绩更新
        self.elitepolicy()

    def solve(self):
        result = []
        BESTFOOD = self.bestFood
        BESTSCORE = calculateFitness(self.bestFood)
        for k in range(self.maxIterCount):
            self.employedBeePhase()
            self.onlookerBeePhase()
            self.scoutBeePhase()
            bestscore = calculateFitness(self.bestFood)  # 当前最优
            # self.bestFood = self.foodList[np.argmax(self.foodScore)]
            if BESTSCORE < bestscore:
                BESTFOOD = self.bestFood
                BESTSCORE = bestscore
            x = (1-BESTSCORE)/BESTSCORE
            result.append(x)
        np.save('IIIABC_result', result)
        BEST_seq = seq_chrom(BESTFOOD)
        print(BEST_seq)
        print((1-BESTSCORE)/BESTSCORE)
        self.printResult(np.array(result))

    def printResult(self, trace):
        x = np.arange(trace.shape[0])
#        GA = np.load('result_ga.npy')
        plt.plot(x, [d for d in trace[:]], 'r', label='optimal value')
#        plt.plot(x, GA)
#        plt.plot(x, [(1-d)/d for d in trace[:, 1]], 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Artificial Bee Swarm algorithm for function optimization")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # random.seed(222)
    # np.random.seed(222)
    t1 = time.time()
    vardim = height
    choromosome_length = height
    is_crossover = 1  # 指定是否交叉
    pc = 1.0
    num = 30
    invalidCount = np.zeros(num)
    bound = np.tile([[0], [1]], vardim)
    abs = ArtificialBeeSwarm(num, num, bound, 600, 100)
    abs.solve()
    t2 = time.time()
    print('total time cost:{}'.format(t2-t1))
