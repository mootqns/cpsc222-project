import pandas as pd

def load_data(csv):
    df = pd.read_csv(csv)
    return df

def clean_titles(ser1, ser2):
    for i in range(0, len(ser1), 1):
        curr = str(ser1[i])
        if("Avatar" in curr):
            ser2[i] = "Avatar: The Last Airbender"
        else:
            cleaned_title = curr.split(":", 1)
            ser2[i] = cleaned_title[0]

    return ser2

def stats(ser1, ser2, df):

    ## duration stats
    mean_duration = ser1.mean()
    total_mins_watched = ser1.sum()
    frequent_duration = ser1.mode()

    # release yr stats
    avg_release_yr = ser2.mean()
    frequent_release_yr = ser2.mode()

    # total number watched
    num_watched = len(df)

    # build stats ser
    stats_list = [mean_duration, frequent_duration[0],frequent_release_yr[0], round(avg_release_yr,3), total_mins_watched, num_watched]
    stats_index = ["Average Duration", "Most Frequent Duration", "Most Frequent Release Year", "Average Release Year", 
    "Total Minutes Watched", "Number Watched", ]
    stats_ser = pd.Series(stats_list, index=stats_index)
    print(stats_ser)
    print()

def type_stats(df, str):
    type_count = df.groupby([str])[str].count()
    ratio =((type_count/len(df))).round(3)
    ratio_df = pd.DataFrame(ratio)
    print(ratio_df)
    print()

    return ratio_df




