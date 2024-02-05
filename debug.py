import os
import pandas as pd

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def read_2021():
    df = pd.read_spss(os.path.join("data", "2021.sav"), convert_categoricals=False)
    df["year"] = 2021
    return df.copy()


def reead_upto2018():
    df = pd.read_spss(os.path.join("data", "upto2018.sav"), convert_categoricals=False)
    return df.copy()


df_2021 = read_2021()  # pd.read_spss(os.path.join("data", "2021.sav"), convert_categoricals=False)
df_upto2018 = reead_upto2018()  # pd.read_spss(os.path.join("data", "upto2018.sav"), convert_categoricals=False)

print(df_2021.columns.to_list())
print(df_2021["wghtpew"])

print(pd.concat([df_upto2018, df_2021], ignore_index=True).reset_index(drop=True)["wghtpew"])
