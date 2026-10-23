import pandas as pd
from datetime import timedelta


DATA_INPUT_DIR = "./data/original/{name}_hourly.csv"
DATA_CLEANED_DIR = "./data/cleaned/{name}_hourly_CLEANED.csv"
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
DATE_REGEX = r"^\d{4}-\d{2}-\d{2} \d{2}:00:00$"  # YYYY-MM-dd HH:00:00

def read_file_and_rename_megawatts_col(dictionary: dict[str, pd.DataFrame]):
    for file in DATA_FILENAMES:
        df = pd.read_csv(DATA_INPUT_DIR.format(name=file))
        columns = [DATETIME, MEGAWATTS]
        df.columns = columns
        dictionary[file] = df
    return dictionary

def drop_duplicates(df):
    #print(f"Missing Values:\n{df.isna().sum()}\n") # missing
    duplis = df[df.duplicated(keep="first")]
    print(f"DUPLICATES: {len(duplis.index)}")
    print(f"FOUND:\n{duplis}")
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index(inplace=True, drop=True)
    if "index" in df.columns: df.drop(columns="index", inplace=True)
    isRemoved = False
    try:
        f = df.loc[duplis.index]
    except KeyError:
            isRemoved = True

    print(f"Has been removed: {isRemoved}")

def drop_rows_null(df):
    mask= df.isna()
    tmp = df.loc[mask.values==True]
    print(f"\nFound {len(tmp.index)} rows w/ NaN/None/Null data:\n{tmp}\n")
    df.dropna(inplace=True)

def drop_cols_null(df):
    old_cols = df.columns.array
    threshold = round(len(df.index) * 0.3) # threshold: 30% of the column values are null
    df.dropna(
        axis=1,
        thresh= threshold,
        inplace=True
    )
    new_cols = df.columns.array
    diff = list(set(old_cols) - set(new_cols))
    print(f"Removed Column: {diff}")

def format_dates(df):
    # attempts to transform all dates into dates, if it fails it replaces it with NaT (Not a Time)
    parsed = pd.to_datetime(df[DATETIME], errors="coerce")
    # mask of NaT values
    mask = parsed.isna()
    # rows w/ NaT as dates
    not_a_time = df[mask.values == True]

    # successfully converted rows
    true_dates = df[mask.values == False]
    s = true_dates[DATETIME].astype("string")

    # bad--> datetimes that aren't a full hour
    bad = ~s.str.match(DATE_REGEX, na=False)
    not_full = true_dates[bad.values == True]
    tmp = not_full[DATETIME].astype(str).str.split(":")

    g_i = list(set(true_dates.index) - set(tmp.index))
    good_dates = true_dates.loc[g_i]
    good_dates[DATETIME] = pd.to_datetime(good_dates[DATETIME])

    # Not a full hour handling
    for i in tmp.index:
        truncated_date = pd.Timestamp(f"{tmp.loc[i][0]}:00:00")

        f = good_dates[good_dates[DATETIME].dt.hour == truncated_date.hour]
        mean = good_dates.loc[f.index, MEGAWATTS].mean()

        tmp.loc[i] = truncated_date
        data = {
            DATETIME: truncated_date,
            MEGAWATTS: mean
        }
        df.loc[i] = data

    # NaT handling, tries two times, if it doesn't succeed by then it drops the row
    for i in not_a_time.index:
        below = -1
        try:
            below = i - 1
            df.loc[below]
        except KeyError:
            try:
                below = i + 1
                df.loc[below]
            except KeyError:
                raise Exception(f"Failed to get above/below: {i}")
        finally:
            data = {}
            isValidDate = False
            if below in range(0, len(df.index)):
                belowDate = pd.to_datetime(df.loc[below, DATETIME])
                newDate = belowDate + timedelta(hours=1)
                #print(newDate)
                #print(newDate in df[DATETIME].values)
                if newDate not in df[DATETIME].values:
                    data[DATETIME] = newDate
                    isValidDate = True
                else:
                    newDate = pd.to_datetime(df.loc[below + 2, DATETIME]) + timedelta(hours=1)
                    if newDate not in df[DATETIME].values:
                        data[DATETIME] = newDate
                        isValidDate = True
                if isValidDate:
                    filtered = parsed[parsed.dt.hour == newDate.hour]
                    mean = df.loc[filtered.index, MEGAWATTS].mean()
                    data[MEGAWATTS] = mean
                    df.loc[i] = data
                    print(f"Added:\n{df.loc[i]}")
                else:
                    print(f"Dropped:\n{df.loc[i]}")
                    df.drop(index=i, inplace=True)
            df[DATETIME] = pd.to_datetime(df[DATETIME], errors="coerce")
            print(df[df[DATETIME].isna().values == True])
            print(df.info())

def save_cleaned_data(dictionary: dict[str, pd.DataFrame]):
    for file in DATA_FILENAMES:
        df = dictionary[file]
        columns = [DATETIME, f"{file}_{MEGAWATTS}"]
        df.columns = columns
        df.to_csv(DATA_CLEANED_DIR.format(name=file))
    return dictionary


def main():
    data_frames: dict[str, pd.DataFrame] = {}
    data_frames = read_file_and_rename_megawatts_col(data_frames)

    strings = []

    for key in data_frames:
        df = data_frames[key]
        num_rows_b = len(df.index)
        num_cols_b = len(df.columns)


        drop_duplicates(df)
        drop_rows_null(df)
        drop_cols_null(df)
        format_dates(df)

        num_rows_a = len(df.index)
        num_cols_a = len(df.columns)

        dropped_rows = num_rows_b - num_rows_a
        dropped_cols = num_cols_b - num_cols_a

        dropped_stats = f"{key}\nNumber of Rows Before Cleaning: {num_rows_b}\nNumber of Rows After Cleaning: {num_rows_a}\nDropped Rows: {dropped_rows} ({float(dropped_rows)/float(num_rows_b)})\n"
        dropped_stats += f"\nNumber of Columns Before Cleaning: {num_cols_b}\nNumber of Columns After Cleaning: {num_cols_a}\nDropped Columns: {dropped_cols} ({float(dropped_cols)/float(num_cols_b)})\n"

        statistics = df.describe().round(3)

        dropped_stats += f"{statistics}"
        strings.append(dropped_stats)
        with open(f"./data/stats_text/{key}_stats.txt", 'w') as output:
            output.write(dropped_stats)

    save_cleaned_data(dictionary=data_frames)

    for stats in strings:
        print(f"\n----------------------\n{stats}\n----------------------\n")

if __name__ == "__main__":
    main()
