import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.distributions.empirical_distribution import ECDF
import random
import market_generator


class GenDic:

    def __init__(self, gen_method_, all_index_, new_df_, scenerioLen_, nTicker_):
        self.__gen_method = gen_method_
        self.__all_index = all_index_
        self.__new_df = pd.DataFrame(new_df_.copy())
        self.__scenerioLen = scenerioLen_
        self.__nTicker = nTicker_
        self.__Retdic = {}
        self.__process_df()
        self.__GenHistDic()
        self.__Gen_dic()
        self.__sDicHist = syn_dic(self.__histDic)
        self.__sDicGen = syn_dic(self.__Retdic)

    def __Gen_dic(self):
        self.__GenMethod = [self.__R_1, self.__R_2, self.__R_3, self.__S]
        if(self.__gen_method == "R1"):
            self.__generate_dic_sub(self.__GenMethod[0])
        elif(self.__gen_method == "R2"):
            self.__generate_dic_sub(self.__GenMethod[1])
        elif(self.__gen_method == "R3"):
            self.__generate_dic_sub(self.__GenMethod[2])
        elif(self.__gen_method == "S"):
            self.__generate_dic_sub(self.__GenMethod[3])
        elif(self.__gen_method == "ML"):
            self.__GenMLdic()

    def __GenMLdic(self):
        key_list = self.__histDic.keys()
        ret = {}
        for key in key_list:
            ret[key] = pd.DataFrame(index=self.__histDic[key].index)
        for ticker in self.__new_df.columns[:self.__nTicker]:
            all_ts = list()
            for key in key_list:
                all_ts.append(np.array(self.__histDic[key][ticker]))
            all_ts = np.array(all_ts)
            MG = market_generator.MarketGenerator("abc", all_ts)
            MG.train(n_epochs=1000)
            generated = np.array([MG.generate(cond) for cond in MG.conditions])
            generated = list(generated)
            count = 0
            for key in key_list:
                ret[key][ticker] = generated[count]
                count += 1
        self.__Retdic = ret

    def __process_df(self):
        self.__new_df["index_number"] = list(range(self.__new_df.shape[0]))

    def __GenHistDic(self):
        dic = {}
        for i in range(len(self.__all_index)):
            row_number = self.__new_df.loc[self.__all_index[i], 'index_number']
            temp_data = self.__new_df.iloc[row_number - int(self.__scenerioLen/2):row_number + int(self.__scenerioLen/2), :self.__nTicker]
            dic[str(self.__all_index[i])] = temp_data

        self.__histDic = dic

    def __generate_dic_sub(self, gen_method):
        dic = {}
        for i in range(len(self.__all_index)):
            row_number = self.__new_df.loc[self.__all_index[i], 'index_number']
            temp_data = self.__new_df.iloc[row_number - int(self.__scenerioLen/2): row_number + int(self.__scenerioLen/2), :self.__nTicker]
            dic[str(self.__all_index[i])] = gen_method(temp_data)
        self.__Retdic = dic

    def __R_1(self, data):
        new_scenerio = pd.DataFrame()
        for i in data.columns:
            ecdf = ECDF(list(data[i]))
            gen = list()
            for j in range(self.__scenerioLen):
                x = np.random.uniform(0, 1, 1)[0]
                gen.append(self.__helper_R_1(x, ecdf.y, ecdf.x))
            new_scenerio[i] = gen
        return new_scenerio

    def __helper_R_1(self, x, prob, data):
        if(x == 1):
            return data[-1]
        elif(x < prob[1]):
            return data[1]
        left = 0
        right = len(data)
        mid = (left + right) // 2
        while(right - left > 1):
            if x < prob[mid]:
                right = mid
            elif x > prob[mid]:
                left = mid
            else:
                return data[mid]
            mid = (left + right) // 2
        return (data[left] + (data[right] - data[left]) * (x - prob[left]) / (prob[right] - prob[left]))

    def __R_2(self, data):
        mean = np.array(data.mean(axis=0))
        cov = data.cov()
        new_scenerio = pd.DataFrame(np.random.multivariate_normal(mean, cov, self.__scenerioLen), columns=data.columns)
        return new_scenerio

    # now p = 0.2
    def __R_3(self, data):
        new_scenerio = pd.DataFrame()
        for i in data.columns:
            new_scenerio[i] = self.__helper_R_3(list(data[i]))
        return new_scenerio

    def __helper_R_3(self, data):
        ret = list()
        last_rand = random.randrange(len(data))
        ret.append(data[last_rand])
        for i in range(1, len(data)):
            r_or_c = random.random()
            if r_or_c < 0.2 and last_rand < len(data) - 1:
                last_rand += 1
            else:
                last_rand = random.randrange(len(data))
            ret.append(data[last_rand])
        return ret

    # var here need attention
    def __S(self, data):
        temp_df = pd.DataFrame(data.copy())
        return_df = pd.DataFrame()
        temp_df['time'] = list(np.arange(1, temp_df.shape[0]+1))
        temp_df['time_square'] = temp_df['time'] ** 2
        X = temp_df[['time', 'time_square']]
        for i in data.columns:
            y = temp_df[i]
            reg = LinearRegression().fit(X, y)
            fitted_value = np.array(temp_df['time']) * reg.coef_[0] + np.array(temp_df['time_square']) * reg.coef_[1] + reg.intercept_
            error_term = y - fitted_value

            err_mu = np.mean(error_term)
            err_var = np.var(error_term, ddof=1)
            new_err = np.random.normal(err_mu, err_var * 30, len(error_term))

            new_ret = fitted_value + new_err
            return_df[i] = new_ret

        return return_df

    def GetGenDic(self):
        return self.__Retdic

    def GetHistDic(self):
        return self.__histDic

    def GetSynGenDic(self):
        return self.__sDicGen

    def GetSynHistDic(self):
        return self.__sDicHist

    def GetMethod(self):
        return self.__GenMethod


def syn_dic(dic_c):
    ret = {}
    key_list = list(dic_c.keys())
    for key in key_list:
        ret[key] = helper_syn_dic(dic_c[key])
    return ret


def helper_syn_dic(dataframe):
    temp = pd.DataFrame(columns=dataframe.columns)
    for col in dataframe.columns:
        temp[col] = np.exp(np.cumsum([0] + list(dataframe[col])))
    temp_2 = list()
    for i in range(1, temp.shape[0]):
        temp_2.append(np.log(sum(temp.iloc[i, :]) / sum(temp.iloc[i-1, :])))
    ret = pd.DataFrame(index=dataframe.index)
    ret["Synthetic"] = temp_2
    return ret
