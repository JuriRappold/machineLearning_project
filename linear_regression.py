from math import sqrt

import pandas as pd
from datetime import timedelta, datetime
"""
epoch = datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0

def gradient_descent(m_now, b_now, x: list, y: list, learn_rate):
    m_gradient = 0
    b_gradient = 0
    if len(x) != len(y): raise Exception("feature and label length are not equal")
    else:
        n = len(x)

        for i in range(n):
            x_i = x[i]#pd.to_datetime(df.loc[i, features], errors="coerce").toordinal()
            y_i = y[i]#df.loc[i, label]

            m_gradient += x_i * (y_i-(m_now * x_i+ b_now))
            b_gradient += (y_i-(m_now * x_i+ b_now))

        print(m_gradient)
        print(b_gradient)
        m = round(m_now - learn_rate * (-(2/n) * m_gradient), 3)
        b = round(b_now - learn_rate * (-(2/n) * b_gradient), 3)

        print(m, b)
        return m, b



def calculate_mean(numbers: list):
    n = len(numbers)
    return sum(numbers) / n

def calculate_variance(numbers, mea):
    n = len(numbers)

    return sum([(i-mea)**2 for i in numbers]) / n

def calculate_standard_deviation(numbers, m):
    return sqrt(calculate_variance(numbers, m))

def normalize_value(xi, mean, std):
    return (xi-mean) / std

def normalize_list(numbers):
    mean = calculate_mean(numbers)
    std = calculate_standard_deviation(numbers, mean)

    normies = [normalize_value(i, mean, std) for i in numbers]

    return normies



today = datetime.now()
toordinal = today.toordinal()
an_extra_hour = today + timedelta(hours=1)
aeh_ordinal = an_extra_hour.toordinal()
print(type(today))
print(f"Dates: {today} & {an_extra_hour}")
print(f"Ordinals: {toordinal} & {aeh_ordinal}")
print(f"Ordinals are equal: {toordinal == aeh_ordinal}")


today_epoch = unix_time_millis(today)
aeh_epoch = unix_time_millis(an_extra_hour)

print(f"\nToday Epoch: {today_epoch}")
print(f"AEH Epoch: {aeh_epoch}")
print(f"Epochs are equal: {today_epoch == aeh_epoch}")
#df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df_agg = df.groupby("datetime")["MW"].sum().reset_index()
df_agg = df.sort_values("datetime").reset_index(drop=True)


# Feature Extraction

1. trend (requires start_date (earliest date in list)


"""
FILE = "./data/all_data_outliers_removed_and_duplicates_averaged.csv"
df = pd.read_csv(FILE)

null_mask = df.isna()

faulty_rows = df.dropna()

print(faulty_rows)
"""
print(df.columns)
df = df.loc[:100]
def to_ordinal(date):
    return date.toordinal()

#dates = list(df["Date"])
dates = list(map(to_ordinal, df["Date"].to_list()))

normalized_dates = normalize_list(dates)
print(normalized_dates[0])



m = 0
b = 0
L = 0.0001 # learn rate
epochs = 300 # increase this for the actual training

for i in range(epochs):
    print(f"on run nr: {i}")

    #
    m, b = gradient_descent(
        m_now= m,
        b_now= b,
        x= dates,
        y=df["MW"].to_list(),
        learn_rate=L,
        #features="Date",
        #label="MW"
    )
    #
print(f"M: {m}\nb: {b}")
"""

