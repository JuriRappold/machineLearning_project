import pandas as pd

df = pd.read_csv("../data/all_data_two.csv")
DATA_CLEANED_DIR = "../data/{name}.csv"
file = "all_data_two_CLEANED"

# Columns
DATETIME = "Datetime"
MEGAWATTS= "MW"
COMPANY = "Company"

DROP_COUNTER = 0
ALL_ROWS_COUNT = len(df)
DATE_REGEX = r"^\d{4}-\d{2}-\d{2} \d{2}:00:00$"  # YYYY-MM-dd HH:00:00
# 1. load files & convert MW column name

drop_counter = 0

# 2. drop duplicate rows, keep first instance
before_indexes = df.index

duplis = df[df.duplicated(subset=[DATETIME, COMPANY], keep="first")]
print(f"DUPLICATES: {len(duplis.index)}")
print(f"FOUND:\n{duplis}")

df.drop_duplicates(subset=[DATETIME, COMPANY], keep='first', inplace=True)

dropped_indexes = list(set(before_indexes) - set(df.index))
df.reset_index(drop=True, inplace=True)
drop_counter += len(dropped_indexes)




# 3. Convert the Datetime column to Datetime objects
df[DATETIME] = pd.to_datetime(df[DATETIME], errors="coerce")

# 4. Drop all Rows w/ None/null values (Datetime == NaT or MW == NaN)
before_indexes = df.index

mask= df.isna()
tmp = df.loc[mask.values==True]
print(f"\nFound {len(tmp.index)} rows w/ NaN/None/Null data:\n{tmp}\n")
#df.dropna(inplace=True)

dropped_indexes = list(set(before_indexes) - set(df.index))
df.reset_index(drop=True, inplace=True)
drop_counter += len(dropped_indexes)

# 6. Check Time of Datetime: Not in format `HH:00:00`? --> drop
before_indexes = df.index

s = df[DATETIME].astype(str)
not_full_hour = ~s.str.match(DATE_REGEX, na=False)

bad_time = df[not_full_hour.values == True]
print(f"\nFound {len(bad_time)} rows w/ bad times:\n{bad_time}")
#df.drop(index=df[not_full_hour.values == True].index, inplace=True)

dropped_indexes = list(set(before_indexes) - set(df.index))
df.reset_index(drop=True, inplace=True)
drop_counter += len(dropped_indexes)


# 5. Check Date of Datetime: Not all 24h in a day? --> drop
before_indexes = df.index

full_days = df.groupby(df[DATETIME].dt.date)[DATETIME].apply(
    lambda x: x.dt.hour.nunique() == 24
)
mask = df[DATETIME].dt.date.map(full_days)

dropped_rows = df[~mask]
print(f"\nFound {len(dropped_rows)} rows w/ not all 24h in a day:\n{dropped_rows}")
#indexes = dropped_rows.index
#df.drop(index= indexes, inplace=True)
dropped_indexes = list(set(before_indexes) - set(df.index))
#print(f"Not full 24h matches: {len(indexes) == len(dropped_indexes)}")
df.reset_index(drop=True, inplace=True)
drop_counter += len(dropped_indexes)



# 7. Check the company column for correctness:
before_indexes = df.index

companies = [
    "AEP",
    "COMED",
    "DAYTON",
    "DEOK",
    "DOM",
    "DUQ",
    "EKPC",
    "FE",
    "NI",
    "PJM_Load",
    "PJME",
    "PJMW"
]
good =  df[COMPANY].isin(companies)

tmp = df[~good].index

df.drop(index=tmp, inplace=True)


dropped_indexes = list(set(before_indexes) - set(df.index))
df.reset_index(drop=True, inplace=True)
drop_counter += len(dropped_indexes)




# Done
DROP_COUNTER += drop_counter
print(f"Dropped {DROP_COUNTER} rows in total: {DROP_COUNTER} / {ALL_ROWS_COUNT} ({(DROP_COUNTER / ALL_ROWS_COUNT) * 100:.2f}%)")
df.to_csv(DATA_CLEANED_DIR.format(name=file))
print(df.columns)