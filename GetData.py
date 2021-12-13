#Author: Hanwen Zhang

import numpy as np
import pandas as pd
import glob


class GetData:

    def __init__(self, window_size_, half_life_, dir_) -> None:
        self.__window_size = window_size_
        self.__half_life = half_life_
        self.__dir = dir_
        self.__load_data()
        self.__new_df = pd.DataFrame(self.__pure_returns.copy())
        self.__number_of_index = self.__pure_returns.shape[1]
        self.__add_rolling()
        self.__new_df["index_number"] = list(range(self.__new_df.shape[0]))
        self.__new_df = self.__new_df.dropna()

    def __load_data(self):
        self.__pure_returns = pd.DataFrame()
        for file_name in glob.glob(self.__dir+'*.csv'):
            x = pd.read_csv(file_name)
            ticker = file_name[len(self.__dir):-4]
            new_add = self.__load_data_sub(x, ticker)
            self.__pure_returns = \
                pd.concat([self.__pure_returns, new_add], axis=1)
        self.__pure_returns = self.__pure_returns.sort_index().dropna()

    def __load_data_sub(self, data, ticker):
        temp = data.copy()
        temp["Date"] = pd.to_datetime(temp['Date']).dt.date
        temp = temp.set_index('Date')
        temp[ticker + "_Return"] = np.log(temp.Price) \
            - np.log(temp.Price.shift(1))
        temp = temp.drop(columns=['Price'])
        temp = temp.dropna()
        return temp

    # this function is helper function to compute rolling  z-score
    def __compute_w_m_sigma(self, X, weights):
        mean = np.ma.average(X, axis=0, weights=weights)
        mean = pd.Series(mean, index=list(X.keys()))
        xm = X - mean
        xm = xm.fillna(0)
        sigma = 1./(weights.sum()-1) * xm.mul(weights, axis=0).T.dot(xm)
        return np.array(mean), np.array(sigma)

    # this function is helper function to compute rolling  z-score
    def __exp_weight(self):
        cur_index = 0
        result = []
        cur_prob = 1
        while cur_index < self.__window_size:
            cur_prob = cur_prob / 2
            add = cur_prob / self.__half_life
            result.extend([add] * self.__half_life)
            cur_index += self.__half_life
        ret = result[:self.__window_size]
        ret.reverse()
        return np.array(ret)

    def __add_rolling(self):
        e_weight = self.__exp_weight()
        self.__new_df["rolling_z"] = np.nan
        for i in range(self.__window_size, self.__pure_returns.shape[0]):
            temp_data = self.__new_df.iloc[i-self.__window_size:i, :self.__number_of_index]
            mean, sigma = self.__compute_w_m_sigma(temp_data, e_weight)
            inver_sigma = np.linalg.inv(sigma)
            current = np.array(self.__new_df.iloc[i, :self.__number_of_index])
            diff = current - mean
            self.__new_df['rolling_z'][i] = - (np.dot(np.dot(diff, inver_sigma), diff.reshape(len(diff), 1)))[0] / np.sqrt(self.__number_of_index)

    def GetNewDF(self):
        return self.__new_df

    def GetNIndex(self):
        return self.__number_of_index

    def GetPureReturns(self):
        return self.__pure_returns
