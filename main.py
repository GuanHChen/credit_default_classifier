import pandas as pd
cc = pd.read_csv("UCI_Dataset.csv", index_col=0)
print(cc.describe())
print(cc.columns)


