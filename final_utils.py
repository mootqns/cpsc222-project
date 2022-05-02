import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as statss
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

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
    print()
    return ratio_df

def pie_chart(df):
    colors = ("#fca78f", "#ead1dc", "#A7C7E7","#D3E19F")
    wp = { 'linewidth' : 0.75, 'edgecolor' : "black" }
    explode = (0.06, 0.05, 0.075, 0.1)

    plt.figure(figsize =(10, 9))
    plt.pie(df["Type"], labels = df.index, colors=colors, wedgeprops=wp, explode=explode)
    plt.title("Netflix Watch Type Ratios")
    plt.tight_layout()
    plt.show()

def pie_chart1(df):
    colors = ("#fca78f","#D3E19F", "#fce9db","#ead1dc", "#bcbcbc", "#A7C7E7")
    wp = { 'linewidth' : 0.75, 'edgecolor' : "black" }
    explode = (0.075, 0.0, 0.075, 0.0, 0.075, 0.0)

    plt.figure(figsize =(10, 9))
    plt.pie(df["Maturity Rating"], labels = df.index, wedgeprops=wp, colors=colors, explode=explode)
    plt.title("Netflix Maturity Rating Ratios")
    plt.tight_layout()
    plt.show()

def barh_chart(df):
    sum_dur_list = df.groupby(['Title'], as_index=False)['Duration'].sum()

    plt.figure(figsize =(10, 7))
    plt.barh(sum_dur_list['Title'], sum_dur_list['Duration'], color='#D3E19F')
    plt.xlabel("Total Minutes Watched")
    plt.ylabel("Titles")
    plt.title("Total Duration for Each Title")
    plt.tight_layout()
    plt.show()

def bar_chart(df):
    weekday_df = df[~df['Weekday'].isin(['Saturday', 'Sunday'])]
    weekend_df = df[~df['Weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]

    weekday_total_mins = weekday_df['Duration'].sum()
    weekend_total_mins = weekend_df['Duration'].sum()

    plt.figure(figsize=(10,8))
    plt.bar(["Weekday", "Weekend"], [weekday_total_mins, weekend_total_mins], color = ["#D3E19F", '#ead1dc'])
    plt.xlabel("Day Type")
    plt.ylabel("Minutes Spent Watching")
    plt.title("Day Type & Total Minutes Spent Watching")
    plt.tight_layout()
    plt.show()

def ind_t_test(df1, df2):
    t_computed, p_value = statss.ttest_ind(df1["Duration"], df2["Duration"])
    print("t-computed:", t_computed, "\np-value:", p_value/2)

def dep_t_test(df1, df2, str):
    t_computed, p_value = statss.ttest_rel(df1[str], df2[str])
    print("t-computed:", t_computed, "\np-value:", p_value/2)

def label_encoder(df, str):
    le = LabelEncoder()
    le.fit(df[str])
    df[str] = le.transform(df[str])

    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping)

def kNN(df, str):
    X = df.drop(str, axis = 1)
    y = df[str].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    knn_clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn_clf.fit(X_train, y_train)

    accuracy = knn_clf.score(X_test, y_test)
    print("accuracy = ", accuracy)

def tree(df, str):
    X = df.drop(str, axis = 1)
    y = df[str].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print("accuracy = ", accuracy)