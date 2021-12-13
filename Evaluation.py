#Author: Hanwen Zhang


import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
from tqdm import tqdm
from utils.leadlag import leadlag
import process_discriminator
import iisignature


class Evaluation:

    def __init__(self, dic_h, dic_g, Gen_method_):
        self.__dic_hist = dic_h
        self.__dic_gen = dic_g
        self.__Gen_method = Gen_method_
        self.__key_list = list(dic_h.keys())
        self.__columns = dic_h[self.__key_list[0]].columns
        self.__compute_dic_diff()
        self.__compute_corr_diff_dic()

    def __compute_stats(self, dataframe):
        ret = pd.DataFrame(columns=self.__columns, index=["Mean", "Variance", "Skewness", "Kurtosis", "Min", "Max"])
        for i in dataframe.columns:
            temp_list = []
            temp_list.append(np.mean(dataframe[i]))
            temp_list.append(np.var(dataframe[i]))
            temp_list.append(skew(dataframe[i]))
            temp_list.append(kurtosis(dataframe[i]))
            temp_list.append(min(dataframe[i]))
            temp_list.append(max(dataframe[i]))
            ret.loc[:, i] = np.round(temp_list, 6)
        return ret

    def __compute_diff(self, dataframe_g, dataframe_h):
        ret = pd.DataFrame(columns=self.__columns, index=dataframe_g.index)
        for i in range(dataframe_g.shape[0]):
            for j in range(dataframe_g.shape[1]):
                if dataframe_h.iloc[i, j] != 0:
                    ret.iloc[i, j] = (dataframe_g.iloc[i, j] - dataframe_h.iloc[i, j]) / dataframe_h.iloc[i, j]
                elif dataframe_g.iloc[i, j] != 0:
                    ret.iloc[i, j] = (dataframe_g.iloc[i, j] - dataframe_h.iloc[i, j]) / dataframe_g.iloc[i, j]
                else:
                    ret.iloc[i, j] = 0
        return ret

    def __compute_dic_diff(self):
        dic = {}
        for key in self.__key_list:
            gen_stats = self.__compute_stats(self.__dic_gen[key])
            hist_stats = self.__compute_stats(self.__dic_hist[key])
            cur_diff = self.__compute_diff(gen_stats, hist_stats)
            dic[key] = cur_diff
        self.__dic_diff = dic

    def __compute_avg_diff(self):
        index_list = (['Mean Diff of Mean', "Mean Diff of Var", "Mean Diff of Skew",
                      "Mean Diff of Kurt", "Mean Diff of Min", "Mean Diff of Max"])
        ret = pd.DataFrame(0, columns=self.__columns, index=index_list)
        for key in self.__key_list:
            for i in range(ret.shape[0]):
                for j in range(ret.shape[1]):
                    ret.iloc[i, j] += self.__dic_diff[key].iloc[i, j]
        self.__avg_diff = ret / len(self.__key_list)

    def __compute_std_diff(self):
        index_list = (['Std Diff of Mean', "Std Diff of Var", "Std Diff of Skew", "Std Diff of Kurt",
                      "Std Diff of Min", "Std Diff of Max"])
        ret = pd.DataFrame(0, columns=self.__columns, index=index_list)
        for row in range(ret.shape[0]):
            for col in range(ret.shape[1]):
                temp_list = list()
                for key in self.__key_list:
                    temp_list.append(self.__dic_diff[key].iloc[row, col])
                ret.iloc[row, col] = np.std(temp_list)
        self.__std_diff = ret

    def __compute_corr_diff_dic(self):
        dic = {}
        for key in self.__key_list:
            dic[key] = (self.__dic_gen[key].corr() - self.__dic_hist[key].corr())
        self.__corr_diff_dic = dic

    def __KS_test_helper(self, dataframe_h, dataframe_g):
        ret = list()
        for col in self.__columns:
            temp_h = list(dataframe_h[col])
            temp_g = list(dataframe_g[col])
        ret.append(ks_2samp(temp_h, temp_g)[0])
        return np.array(ret)

    def __compute_two_list(self, index):
        g_list = list()
        h_list = list()
        for key in self.__key_list:
            g_list.append(self.__ret_2_price(self.__dic_gen[key][index]))
            h_list.append(self.__ret_2_price(self.__dic_hist[key][index]))
        return g_list, h_list

    def __ret_2_price(self, r_list):
        r_list = list(r_list)
        return list(np.exp(np.cumsum([0] + r_list)))

    def __AvgToOne(self, dataframe):
        temp = list()
        for row in range(dataframe.shape[0]):
            temp.append(np.mean(dataframe.iloc[row, :]))
        ret = pd.DataFrame(index=dataframe.index)
        ret['Stats'] = temp
        return ret

    def GetAvgDiff(self):
        self.__compute_avg_diff()
        self.__avg_diff = pd.concat([self.__AvgToOne(self.__avg_diff), self.__avg_diff], axis=1)
        return self.__avg_diff

    def GetStdDiff(self):
        self.__compute_std_diff()
        self.__std_diff = pd.concat([self.__AvgToOne(self.__std_diff), self.__std_diff], axis=1)
        return self.__std_diff

    def GetAveCorrDiff(self):
        ret = pd.DataFrame(0, columns=self.__columns, index=self.__columns)
        for key in self.__key_list:
            ret += self.__corr_diff_dic[key]
        ret = ret / len(self.__key_list)
        finalRet = list()
        for row in range(ret.shape[0]):
            for col in range(ret.shape[1]):
                finalRet.append(ret.iloc[row, col])
        return round(sum(finalRet) / (len(finalRet) - ret.shape[0]), 6)

    def GetStdCorrDiff(self):
        ret = pd.DataFrame(0, columns=self.__columns, index=self.__columns)
        for row in range(ret.shape[0]):
            for col in range(ret.shape[1]):
                temp_list = list()
                for key in self.__key_list:
                    temp_list.append(self.__corr_diff_dic[key].iloc[row, col])
                ret.iloc[row, col] = np.std(temp_list)
        finalRet = list()
        for row in range(ret.shape[0]):
            for col in range(ret.shape[1]):
                finalRet.append(ret.iloc[row, col])
        return round(sum(finalRet) / (len(finalRet) - ret.shape[0]), 6)

    def GetKSTestTable(self):
        ret = np.zeros(len(self.__columns))
        for key in self.__key_list:
            ret += self.__KS_test_helper(self.__dic_hist[key], self.__dic_gen[key])
        ret = ret / len(self.__key_list)
        retdf = pd.DataFrame(index=["Ave Test Stats"], columns=self.__columns)
        retdf.iloc[0, :] = ret
        retdf = pd.concat([retdf, self.__AvgToOne(retdf)], axis=1)
        return retdf

    def GetMDTestTable(self, con_level):
        retlist = list()
        for col in self.__columns:
            g_list, h_list = self.__compute_two_list(col)
            order = 4
            sigs1 = np.array([np.r_[1., iisignature.sig(leadlag(p), order)] for p in tqdm(g_list)])
            sigs2 = np.array([np.r_[1., iisignature.sig(leadlag(p), order)] for p in tqdm(h_list)])
            res = process_discriminator.test(sigs1, sigs2, order=order, compute_sigs=False, confidence_level=con_level)
            retlist.append(res)
        retdf = pd.DataFrame()
        retdf["Number of T"] = [sum(retlist)]
        retdf["Number of F"] = [len(retlist) - sum(retlist)]
        return retdf

    def SynScenerioPlot(self):
        if(len(self.__columns) > 1):
            print("Unable to plot graph.")
            return None
        else:
            plt.figure(figsize=(10, 6), dpi=80)
            for key in self.__key_list:
                plt.plot(self.__ret_2_price(self.__dic_gen[key].iloc[:, 0]), color='b')
                plt.plot(self.__ret_2_price(self.__dic_hist[key].iloc[:, 0]), color='g')
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.annotate('* Generated', xy=(0.05, 0.95), xycoords='axes fraction', color='b', fontsize=13)
            plt.annotate('* Historical', xy=(0.05, 0.90), xycoords='axes fraction', color='g', fontsize=13)
            plt.show()
            return None

    def GetGenMethod(self):
        return self.__Gen_method
