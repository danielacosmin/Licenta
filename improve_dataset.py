import pandas as pd
import pprint
from scipy import stats
import numpy as np 

data = pd.read_csv("dataset2.csv")
print(data.shape)

def split_dataset_by_mood(mood):
    mood_cond = data.mood.str.contains(mood)
    df_split = data[mood_cond]
    return df_split

def remove_outliers_zscore(df):

    z = np.abs(stats.zscore(df.iloc[:,5:-1]))
    threshold = 3
    outliered_df = df[(z < 3).all(axis=1)]
    return outliered_df
def main():

    happy_df = split_dataset_by_mood("Happy")
    # # print(happy_df.shape)
    happy_df_out = remove_outliers_zscore(happy_df)
    # # print(happy_df_out.shape)
    # # print(happy_df_out.head)
    # happy_df_out.to_csv("happy_outline.csv", index = false)


    sad_df = split_dataset_by_mood("Sad")
    # print(sad_df.shape)
    sad_df_out = remove_outliers_zscore(sad_df)
    # print(sad_df_out.shape)
    # print(sad_df_out.head)

    energy_df = split_dataset_by_mood("Energetic")
    # print(energy_df.shape)
    energy_df_out = remove_outliers_zscore(energy_df)
    # print(energy_df_out.shape)
    # print(energy_df_out.head)

    chill_df = split_dataset_by_mood("Calm")
    # print(chill_df.shape)
    chill_df_out = remove_outliers_zscore(chill_df)
    # print(chill_df_out.shape)
    # print(chill_df_out.head)

    final_df = pd.concat([happy_df_out, sad_df_out, energy_df_out, chill_df_out], ignore_index = True, sort = False)
    print(final_df.shape)
    final_df.to_csv("outliered.csv", index = False)


if __name__ == "__main__":
    main()