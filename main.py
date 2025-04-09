import pandas as pd
import statistics as s
from sklearn.preprocessing import OneHotEncoder

dd = pd.read_csv("UCI_Dataset.csv", index_col=0)

new_cols = ['lim', 'sex', 'edu', 'mar', 'age', 'repay1', 'repay2', 'repay3', 'repay4',
            'repay5', 'repay6', 'bill1', 'bill2', 'bill3', 'bill4', 'bill5', 'bill6',
            'pay1', 'pay2', 'pay3', 'pay4', 'pay5', 'pay6', 'default']

dd.columns = new_cols

"""
Binary encoding of SEX variable
Female = 0
Male = 1
"""

dd["sex"] = dd["sex"].replace({2:0})

"""
Ordinal encoding of EDUCATION
0 = Others, Unknown         280+123+51+14
1 = High School             4917
2 = University              14030
3 = Graduate School         10585
"""

mapping = {
    1: 3,  # Graduate school to 3
    2: 2,  # University to 2
    3: 1,  # High school to 1
    4: 0,  # Others to 0
    5: 0,  # Unknown to 0
    6: 0,  # Unknown to 0
    0: 0   # For some reason, there are 14 instances of 0. These are also remapped to 0.
}
dd['edu'] = dd['edu'].map(mapping)

"""
One-hot encoding of MARRIAGE
0 = Not sure what this is   54
1 = Married                 13659
2 = Single                  15964
3 = Other                   323

Single and Married will be one hot encoded
Values of 0 and for Other will be dropped. If someone is neither married or single,
the other category is implicit
"""

enc = OneHotEncoder(sparse_output=False)
enc_mar = enc.fit_transform(dd[['mar']])
enc_col = enc.get_feature_names_out(['mar'])
mar_df = pd.DataFrame(enc_mar, columns=enc_col)
mar_df = mar_df.iloc[:, [1,2]]

# Onehot encoded dataframe

dd_part1 = dd.iloc[:, :4]
dd_part2 = dd.iloc[:, 4:]
dd = pd.concat([dd_part1, mar_df, dd_part2], axis=1)
dd.drop(columns = 'mar', inplace = True)
dd = dd.rename(columns={'mar_1': 'married', 'mar_2': 'single'})

# Insert the encoded dataframe where the mar variable is
# Drop the mar column

"""
Feature Engineering
Credit utilization
637 Counts have utilization greater than 1
201 Counts have utilization less than 0 
"""

dd['util'] = dd[['bill1', 'bill2', 'bill3', 'bill4', 'bill5', 'bill6']].sum(axis=1) / (6 * dd['lim'])





print(dd['util'].describe())
print((dd['util'] < 0).sum())
#print(dd['mar'].value_counts())



