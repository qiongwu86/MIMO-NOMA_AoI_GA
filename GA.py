import random
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np

class Gene:
    def __init__(self, **data):
        self.__dict__.update(data)          #在__dict__属性中更新data
        self.size = len(data['data'])


class GA:
    def __init__(self, parameter):
        # 交叉率， 变异率， 繁殖代数， 种群大小， 最小值， 最大值
        #parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
        self.parameter = parameter
        self.NGEN = self.parameter[2] #繁殖代数
        self.CXPB = self.parameter[0] #交叉率
        self.MUTPB = self.parameter[1] #变异率
        low = self.parameter[4] #最小值
        up = self.parameter[5] #最大值
        self.bound = []
        self.bound.append(low)
        self.bound.append(up)

        # self.evaluate = evaluate

        pop = []
        for i in range(self.parameter[3]): #生成popsize的种群个数
            geneinfo = []
            for pos in range(len(low)):    #生成len(low)维的随机数据作为个体初始值
                # geneinfo.append(random.randint(self.bound[0][pos],self.bound[1][pos]))
                geneinfo.append(self.bound[0][pos] + (self.bound[1][pos] - self.bound[0][pos])*random.random())
                
            fitness = self.evaluate(geneinfo)  #计算生成的随机个体的适应度函数值
            pop.append({'Gene':Gene(data=geneinfo), 'fitness':fitness})
        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)   #保存最好的个体数据{'Gene':Gene(), 'fitness':fitness}


    def evaluate(self, geneinfo):
        # 作为适应度函数评估该个体的函数值
        x1 = geneinfo[0]
        x2 = geneinfo[1]
        x3 = geneinfo[2]
        x4 = geneinfo[3]
        y = np.exp(x1 + x2 ** 2) + 100*np.sin(x3 ** 2) + x4**2
        return y

    def selectBest(self, pop):
        #选出当前代种群中的最好个体作为历史记录
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)    #从大到小排列
        return s_inds[0]            #返回依然是一个字典

    def selection(self, individuals, k):
        #用轮盘赌的方式，按照概率从上一代选择个体直至形成新的一代
        #k表示需要选出的后代个数
        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True) #从大到小
        sum_fits = sum(ind['fitness'] for ind in individuals)   #计算所有个体适应度函数的和
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits   #随机产生一个[0,sun_fits]范围的数，即轮盘赌这局的指针
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']       #逐次累加从大到小排列的个体的适应度函数的值，直至超过指针，即选择它
                if sum_ >= u:
                    chosen.append(ind)
                    break
        chosen = sorted(chosen, key=itemgetter('fitness'), reverse=False)  #从小到大排列选择的个体，方便进行交叉操作
        return chosen

    def crossperate(self, offspring):
        #实现交叉操作，交换倆数据随机两维之间的数据
        dim = len(offspring[0]['Gene'].data)  #获得数据维数，即基因位数

        geneinfo1 = offspring[0]['Gene'].data   #交叉的第一个数据
        geneinfo2 = offspring[1]['Gene'].data   #交叉的第二个数据

        if dim == 1:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.randrange(1,dim)
            pos2 = random.randrange(1,dim)

        newoff1 = Gene(data=[])    #后代1
        newoff2 = Gene(data=[])    #后代2
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1,pos2):   #交换的部分维度
                temp1.append(geneinfo2[i])
                temp2.append(geneinfo1[i])
            else:
                temp1.append(geneinfo1[i])
                temp2.append(geneinfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2
        return newoff1, newoff2


    def mutation(self, crossoff, bound):
        #实现单点编译，不实现逆转变异
        dim = len(crossoff.data)

        if dim == 1:
            pos = 0
        else:
            pos = random.randrange(0, dim) #选择单点变异的点

        crossoff.data[pos] = random.randint(bound[0][pos], bound[1][pos])
        crossoff.data[pos] = self.bound[0][pos] + (self.bound[1][pos] - self.bound[0][pos])*random.random()
        return crossoff

    def GA_main(self):
        popsize = self.parameter[3]
        # print('Start of evolution')
        ever_best = []
        for g in range(self.NGEN):
            # print("############ Generation {} ##############".format(g))
            #首先进行选择
            selectpop = self.selection(self.pop, popsize)
            nextoff = []
            while len(nextoff) != popsize:
                offspring = [selectpop.pop() for _ in range(2)] #后代间两两选择
                if random.random() < self.CXPB:       #进行交叉的后代
                    crossoff1, crossoff2 = self.crossperate(offspring)
                    if random.random() < self.MUTPB:
                        muteoff1 = self.mutation(crossoff1, self.bound)
                        muteoff2 = self.mutation(crossoff2, self.bound)
                        fit_muteoff1 = self.evaluate(muteoff1.data)
                        fit_muteoff2 = self.evaluate(muteoff2.data)
                        nextoff.append({'Gene':muteoff1, 'fitness':fit_muteoff1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2})
                    else:
                        fit_muteoff1 = self.evaluate(crossoff1.data)
                        fit_muteoff2 = self.evaluate(crossoff2.data)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_muteoff1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_muteoff2})
                else:
                    nextoff.extend(offspring)   #直接追加两个后代

            self.pop = nextoff   #种群直接变为下一代
            fits = [ind['fitness'] for ind in self.pop]

            best_ind = self.selectBest(self.pop)
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
            ever_best.append(self.bestindividual['fitness'])

            print("Best individual found is {}, {}".format(self.bestindividual['Gene'].data,
                                                           self.bestindividual['fitness']))
            print("  Max fitness of current pop: {}".format(max(fits)))
        plt.plot(ever_best)
        plt.show()
        print('------------ End ----------------')
        return self.bestindividual['Gene'].data
    
if __name__ == '__main__':
    CXPB, MUTPB, NGEN, popsize = 0.8, 0.5, 1000, 1000

    up = [90, 10, 5, 6]
    low = [1, 2, 3, 1]
    parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
    run = GA(parameter)
    run.GA_main()