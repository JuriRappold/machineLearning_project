import pandas as pd
from datetime import timedelta, datetime


DATA_INPUT_DIR = "./data/original/{name}_hourly.csv"
DATA_CLEANED_DIR = "./data/double_checked_hours/{name}_hourly_CLEANED.csv"
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
COLUMNS_TO_DROP = ["index", "level_0"]
# rename MW columns:
DATETIME = "Datetime"
MEGAWATTS= "MW"


def read_file_and_rename_megawatts_col(dictionary: dict[str, pd.DataFrame]):
    for file in DATA_FILENAMES:
        df = pd.read_csv(DATA_INPUT_DIR.format(name=file))
        columns = [DATETIME, MEGAWATTS]
        df.columns = columns
        df[DATETIME] = pd.to_datetime(df[DATETIME], errors="coerce")
        dictionary[file] = df
    return dictionary

def check_hour_range(df: pd.DataFrame):
    dates = df[DATETIME]
    hours = (dates.groupby(dates.dt.date)
             .apply(lambda x: sorted(x.dt.hour.nunique().tolist()) )
             .reset_index(name='Hours')
             .rename(columns={'index': 'Date'})
             )
    hours['Has 24 hours'] = hours['Hours'].apply(lambda h: set(h) == set(range(24)))

    full_days = dates.groupby(dates.dt.date).apply(
        lambda x: set(x.dt.hour) == set(range(24))
    )
    mask = dates.dt.date.map(full_days)
    dropped_indexes = df[mask.values == False]

    #print(dropped_indexes)
    return hours


    #return h

def check_dict_hours(dictionary: dict[str, pd.DataFrame]):
    hours_dict: dict[str, pd.DataFrame] = {}
    for file in DATA_FILENAMES:
        hours_dict[file] = check_hour_range(dictionary[file])
        #check_hour_range(dictionary[file])

    return hours_dict

def main():
    has24 = "Has 24 hours"
    frames = {}

    frames = read_file_and_rename_megawatts_col(frames)

    #unique_hours = check_dict_hours(frames)

    # hours[hours['Has 24 hours'] == False]
    for file in DATA_FILENAMES:
        #values = unique_hours[key]
        #print(f"{values}\tHas 24 entries: {len(values) == 24}\t{key}")
        df = frames[file]

        #old_df = check_hour_range(df)

        #bad_ones = old_df[old_df[has24].values == False]


        full_days = df.groupby(df[DATETIME].dt.date)[DATETIME].apply(
            lambda x: x.dt.hour.nunique() == 24
        )
        mask = df[DATETIME].dt.date.map(full_days)

        dropped_rows = df[~mask]
        print(f"{file} has this many rows that belong to a day without all 24h: {len(dropped_rows)} / {len(df)} ({(len(dropped_rows)/len(df))*100}%)")
        """
        dropped_rows = df.groupby(df[DATETIME].dt.date)[DATETIME].apply(
            lambda x: 
        )

        print(f"df: {len(dropped_rows)}")
        print(f"old_df: {len(bad_ones)}")
        print(f"Are equal: {len(bad_ones) == len(dropped_rows)}")
        #non_24h_days = df[df[has24] == False]
        #print(f"{file} has these non-24h days: {len(non_24h_days)}")
        """


main()



