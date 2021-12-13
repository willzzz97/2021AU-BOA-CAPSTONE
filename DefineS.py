from matplotlib import pyplot as plt
import random
import pandas as pd
import numpy as np


class DefineS:

    def __init__(self, dataframe_, nSpeEvent_, nNormEvent_, minDis_, ScenerioLen_) -> None:
        self.__dataframe = dataframe_
        self.__nSpeEvent = nSpeEvent_
        self.__nNormEvent = nNormEvent_
        self.__minDis = minDis_
        self.__ScenerioLen = ScenerioLen_
        self.__find_irr_index()
        self.__find_normal_year()
        self.__compute_prob()

    def __find_irr_index(self):
        copy_data = list(self.__dataframe["rolling_z"].copy())
        temp_data = list(self.__dataframe["rolling_z"].copy())

        pair = []
        count = 0

        while(count < self.__nSpeEvent and len(temp_data) > 0):
            cur_max = max(temp_data)
            cur_index = copy_data.index(cur_max)
            add = True
            for i in pair:
                if abs(cur_index - i[0]) < self.__minDis:
                    add = False
                    break
            if add:
                pair.append([cur_index, self.__dataframe.iloc[cur_index].name])
                count += 1
            temp_data.remove(cur_max)
        ret = []
        for i in pair:
            ret.append(i[1])
        self.__irr_index = ret

    def __find_normal_year(self):
        num_extreme = list()
        for i in self.__irr_index:
            num_extreme.append(self.__dataframe.loc[i, "index_number"])
        ret = list()
        while(len(ret) < self.__nNormEvent):
            temp = random.randint(int(self.__ScenerioLen/2), self.__dataframe.shape[0] - 1 - int(self.__ScenerioLen/2))
            add = True
            for i in num_extreme:
                if i in range(int(temp - self.__ScenerioLen/2), int(temp + self.__ScenerioLen/2)):
                    add = False
                    break
            if add and (self.__dataframe.iloc[temp, :].name not in ret):
                ret.append(self.__dataframe.iloc[temp, :].name)
        self.__norm_index = ret

    # compute the probability that the special scenerios would happen
    def __compute_prob(self):
        temp_data = list(self.__dataframe["rolling_z"].copy())
        temp_data.sort(reverse=True)
        prob_pair = []
        for i in self.__irr_index:
            cur_z = self.__dataframe["rolling_z"][i]
            prob = (temp_data.index(cur_z) + 1) / len(temp_data)
            prob_pair.append([i, prob])
        self.__probPair = prob_pair

    def ShowGraph(self):
        plt.figure(figsize=(15, 7))
        self.__dataframe.plot(y="rolling_z", figsize=(20, 7), legend=None, title="Rolling Z Score")
        x = []
        y = []
        for i in self.__irr_index:
            x.append(i)
            y.append(self.__dataframe["rolling_z"][i])
        plt.scatter(x, y, c='r')
        plt.xlabel('Date')
        plt.ylabel("rolling_z")
        plt.show()

    def GetNorm(self):
        return self.__norm_index

    def GetIrr(self):
        return self.__irr_index

    def GetAll(self):
        return list(self.__irr_index) + list(self.__norm_index)

    def GetProbPair(self):
        return self.__probPair

    def ShowProbPair(self):
        for i in self.__probPair:
            print('The Secnerio like period around', i[0], 'has the probability of', round(i[1], 6))


def GetProbTable(prob_pairs, pure_returns, s_len):
    ret = pd.DataFrame(columns=["Start Date", "End Date", "Probability"])
    for pair in prob_pairs:
        cur = pair[0]
        start_date = str(pure_returns.iloc[int(np.where(pure_returns.index == cur)[0][0] - s_len / 2), :].name)
        end_date = str(pure_returns.iloc[int(np.where(pure_returns.index == cur)[0][0] + s_len / 2), :].name)
        ret = ret.append({"Start Date": start_date, "End Date": end_date, "Probability": pair[1]}, ignore_index=True)
    ret.index = list(range(1, ret.shape[0]+1))
    return ret
