import get_data as gd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

data_dir = "./Data/"
test = gd.get_data(252, 40, data_dir)

pureReturns = test.GetPureReturns()
dfTable = test.GetNewDF()
nticker = test.GetNIndex()