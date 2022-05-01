import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as statss

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
    print()
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

    return ratio_df

def pie_chart(df):
    colors = ("#cd722d", "#fcd053", "#933639","#D3E19F")
    wp = { 'linewidth' : 0.75, 'edgecolor' : "black" }
    explode = (0.05, 0.075, 0.1, 0.25)

    plt.figure(figsize =(10, 9))
    plt.pie(df["Type"], labels = df.index, colors=colors, wedgeprops=wp, explode=explode)
    plt.title("Netflix Watch Type Breakdown")
    plt.tight_layout()
    plt.show()

def bar_chart(df):
    sum_dur_list = df.groupby(['Title'], as_index=False)['Duration'].sum()

    plt.figure(figsize =(10, 7))
    plt.barh(sum_dur_list['Title'], sum_dur_list['Duration'], color='#D3E19F')
    plt.xlabel("Total Minutes Watched")
    plt.ylabel("Titles")
    plt.title("Total Duration for Each Title")
    plt.tight_layout()
    plt.show()

def ind_t_test(df1, df2):
    t_computed, p_value = statss.ttest_ind(df1["Duration"], df2["Duration"])
    print("t-computed:", t_computed, "\np-value:", p_value/2)
    if(t_computed < 1.697):
        print("Decision:", "fail to reject null")
    if(t_computed > 1.697):
        print("Decision:", "reject null")