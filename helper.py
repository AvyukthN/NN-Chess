import numpy as np

data = np.load('./dataset_1M.npz')

print(data['arr_0'])
print(data['arr_1'][10000])

# import pandas as pd

# df = pd.read_csv('./preprocessed2_data.csv')
# dat = df.iloc[0]

# print(dat)