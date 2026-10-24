"""
1. load files & conver MW column name
2. drop duplicate rows, keep first instance
3. Convert the Datetime column to Datetime objects
4. Drop all Rows w/ None/null values (Datetime == NaT or MW == NaN)
5. Check Date of Datetime: Not all 24h in a day? --> drop
6. Check Time of Datetime: Not in format `HH:00:00`? --> drop
"""

import pandas as pd
from datetime import timedelta


DATA_INPUT_DIR = "../data/original/{name}_hourly.csv"
DATA_CLEANED_DIR = "../data/cleaned_two/{name}_hourly_CLEANED.csv"
DATA_FILENAMES = [
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
# rename MW columns:
DATETIME = "Datetime"
MEGAWATTS= "MW"
DATE_REGEX = r"^\d{4}-\d{2}-\d{2} \d{2}:00:00$"  # YYYY-MM-dd HH:00:00

def main():
    DROP_COUNTER = 0
    ALL_ROWS_COUNT = 0
    #DROPPED_INDEXES: list = []
    #data_frames = {}
    for file in DATA_FILENAMES:
        # 1. load files & convert MW column name
        df = pd.read_csv(DATA_INPUT_DIR.format(name=file))
        columns = [DATETIME, MEGAWATTS]
        df.columns = columns

        original_length = len(df)
        ALL_ROWS_COUNT += original_length

        drop_counter = 0

        # 2. drop duplicate rows, keep first instance
        before_indexes = df.index
        df.drop_duplicates(subset=[DATETIME], keep='first', inplace=True)

        dropped_indexes = list(set(before_indexes) - set(df.index))
        df.reset_index(drop=True, inplace=True)
        drop_counter += len(dropped_indexes)


        # 3. Convert the Datetime column to Datetime objects
        df[DATETIME] = pd.to_datetime(df[DATETIME], errors="coerce")

        # 4. Drop all Rows w/ None/null values (Datetime == NaT or MW == NaN)
        before_indexes = df.index

        df.dropna(inplace=True)

        dropped_indexes = list(set(before_indexes) - set(df.index))
        df.reset_index(drop=True, inplace=True)
        drop_counter += len(dropped_indexes)

        # 6. Check Time of Datetime: Not in format `HH:00:00`? --> drop
        before_indexes = df.index

        s = df[DATETIME].astype(str)
        not_full_hour = ~s.str.match(DATE_REGEX, na=False)

        df.drop(index=df[not_full_hour.values == True].index, inplace=True)

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
        indexes = dropped_rows.index
        df.drop(index= indexes, inplace=True)
        dropped_indexes = list(set(before_indexes) - set(df.index))
        #print(f"Not full 24h matches: {len(indexes) == len(dropped_indexes)}")
        df.reset_index(drop=True, inplace=True)
        drop_counter += len(dropped_indexes)




        # Done
        print(f"Dropped {drop_counter} rows from {file}: {drop_counter} / {original_length} ({(drop_counter/original_length)*100}%)")
        DROP_COUNTER += drop_counter
        df.to_csv(DATA_CLEANED_DIR.format(name=file))
        print(df.columns)

    print(f"Dropped {DROP_COUNTER} rows from: {DROP_COUNTER} / {ALL_ROWS_COUNT} ({(DROP_COUNTER / ALL_ROWS_COUNT) * 100:.2f}%)")



if __name__ =="__main__":
    main()




