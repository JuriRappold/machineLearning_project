import pandas as pd
from check_duplicates import count_duplicate_rows, check_duplicates

def avg_after_relabel():
    df = pd.read_csv("./data/all_data_relabelled.csv")
    # Columns
    DATETIME = "Datetime"
    MEGAWATTS= "MW"
    COMPANY = "Company"

    PJM = "PJM"

    df = df[df["Company"].values == "PJM"]
    if "Unnamed: 0" in df.columns: df.drop(columns="Unnamed: 0", inplace=True)
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
    duplicate_count = count_duplicate_rows(df)#df.duplicated(subset=[DATETIME, COMPANY], keep=False)
    print(f"Should be 0: {duplicate_count}")
    if duplicate_count != 0:
        check_duplicates(df)
        df.drop_duplicates(subset=[DATETIME, COMPANY], keep="first", inplace=True)

    duplicate_count = count_duplicate_rows(df)
    assert(duplicate_count == 0)



    new_length = len(df)
    print(f"Length comparison {original_length} != {new_length}: {original_length != new_length}")
    print("Saving....")
    df.to_csv("./data/cleaned/all_data_relabelled_CLEANED.csv")
    print("Done Saving")

