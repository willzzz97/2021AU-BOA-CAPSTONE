import pandas as pd


class ComputeTable:

    def __init__(self, Eva_list_):
        self.__Eva_list = Eva_list_

    def GetStatsTable_1(self):
        ret = pd.DataFrame(index=self.__Eva_list[0].GetAvgDiff().index)
        for Eva in self.__Eva_list:
            temp_method = Eva.GetGenMethod()
            temp_data = list(Eva.GetAvgDiff().iloc[:, 0])
            ret[temp_method] = temp_data
        return ret

    def GetStatsTable_2(self):
        ret = pd.DataFrame(index=self.__Eva_list[0].GetStdDiff().index)
        for Eva in self.__Eva_list:
            temp_method = Eva.GetGenMethod()
            temp_data = list(Eva.GetStdDiff().iloc[:, 0])
            ret[temp_method] = temp_data
        return ret

    def GetStatsTable_3(self):
        ret = pd.DataFrame(index=['AVG of AVG Corr Diff'])
        for Eva in self.__Eva_list:
            temp_method = Eva.GetGenMethod()
            temp_data = list([Eva.GetAveCorrDiff()])
            ret[temp_method] = temp_data
        return ret

    def GetStatsTable_4(self):
        ret = pd.DataFrame(index=['AVG of STD CORR DIF'])
        for Eva in self.__Eva_list:
            temp_method = Eva.GetGenMethod()
            temp_data = list([Eva.GetStdCorrDiff()])
            ret[temp_method] = temp_data
        return ret

    def GetMDTestTable(self, confi_level):
        ret = pd.DataFrame(index=["# of True", '# of False'])
        for Eva in self.__Eva_list:
            temp_method = Eva.GetGenMethod()
            temp_data = list(Eva.GetMDTestTable(confi_level).iloc[0, :])
            ret[temp_method] = temp_data
        # ret = ret.style.set_caption("MD Test for Confidence Level " + str(confi_level))
        return ret

    def GetKSTable(self):
        ret = pd.DataFrame(index=["KS Test Stats"])
        for Eva in self.__Eva_list:
            temp_method = Eva.GetGenMethod()
            temp_data = list([Eva.GetKSTestTable().iloc[0, -1]])
            ret[temp_method] = temp_data
        return ret
