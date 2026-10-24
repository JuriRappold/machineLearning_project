import pandas as pd


df = pd.read_csv("../data/two/all_data_two_relabelled.csv")
# Columns
DATETIME = "Datetime"
MEGAWATTS= "MW"
COMPANY = "Company"

PJM = "PJM"

df = df[df["Company"].values == "PJM"]
df.drop(columns="Unnamed: 0", inplace=True)
original_length = len(df)
df[DATETIME] = pd.to_datetime(df[DATETIME], errors="raise")
dupli_mask = df.duplicated(subset=[DATETIME, COMPANY], keep=False)

duplis = df[dupli_mask.values == True]

duplis = duplis.sort_values(by=DATETIME)


per_date = duplis.groupby(duplis[DATETIME])[MEGAWATTS].mean()


df.drop_duplicates(subset=[DATETIME, COMPANY], keep="first", inplace=True)

for i in per_date.index:
    print(i)
    df_index = df[(df[COMPANY] == PJM) & (df[DATETIME] == i)].index[0]
    before = df.loc[df_index, MEGAWATTS]

    assert(df.loc[df_index, MEGAWATTS] == before)
    df.loc[df_index, MEGAWATTS] = per_date[i]
    assert (df.loc[df_index, MEGAWATTS] != before)

df.reset_index(drop=True, inplace=True)
duplis_two = df.duplicated(subset=[DATETIME, COMPANY], keep=False)
assert(len(duplis_two) == 0)
print(len(duplis_two) == 0)

new_length = len(df)
print(f"Length comparison {original_length} != {new_length}: {original_length != new_length}")
print("Saving....")
df.to_csv("../data/two/all_data_READY.csv")
print("Done Saving")

