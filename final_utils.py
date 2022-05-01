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

def compute_stats(ser):
    mean = ser.mean()
    sum = ser.sum()
    mode = ser.mode()

    return  mean, sum, mode

def type_stats(df, str):
    type_count = df.groupby([str])[str].count()
    ratio =((type_count/len(df))).round(3)
    ratio_df = pd.DataFrame(ratio)
    print(ratio_df)
    print()

def build_stats_ser(stat1, stat2, stat3, stat4, stat5, stat6):
    stats_list = [stat1, stat3[0], stat5[0], round(stat6,3), stat2, stat4]
    stats_index = ["Average Duration", "Most Frequent Duration", "Most Frequent Release Year", "Average Release Year", 
    "Total Minutes Watched", "Number Watched", ]
    stats_ser = pd.Series(stats_list, index=stats_index)
    print(stats_ser)


