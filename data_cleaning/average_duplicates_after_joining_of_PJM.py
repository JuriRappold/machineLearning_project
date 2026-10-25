import pandas as pd
from check_duplicates import count_duplicate_rows, check_duplicates

def avg_after_relabel():
    df = pd.read_csv("../data/all_data_relabelled.csv")
    # Columns
    DATETIME = "Datetime"
    MEGAWATTS= "MW"
    COMPANY = "Company"

    PJM = "PJM"


    pjm = df.loc[df[COMPANY].eq(PJM)].copy()
    # Save the original PJM index labels before cleaning.
    original_pjm_indexes = df.index[df[COMPANY].eq(PJM)]


    pjm = pjm.drop(columns=["Unnamed: 0"], errors="ignore")
    original_length = len(df)
    pjm[DATETIME] = pd.to_datetime(pjm[DATETIME], errors="raise")
    dupli_mask = pjm.duplicated(subset=[DATETIME, COMPANY], keep=False)

    duplis = pjm[dupli_mask.values == True]

    duplis = duplis.sort_values(by=DATETIME)


    per_date = duplis.groupby(duplis[DATETIME])[MEGAWATTS].mean()


    pjm.drop_duplicates(subset=[DATETIME, COMPANY], keep="first", inplace=True)

    for i in per_date.index:
        print(i)
        df_index = pjm[(pjm[COMPANY] == PJM) & (pjm[DATETIME] == i)].index[0]

        pjm.loc[df_index, MEGAWATTS] = per_date[i]

    print(pjm.index)
    duplicate_count, pjm_cleaned = check_duplicates(pjm, "PJM")
    print(f"Should be 0: {duplicate_count}")

    if duplicate_count:
        assert count_duplicate_rows(pjm_cleaned)[0] == 0
        pjm = pjm_cleaned

    # After averaging and duplicate cleaning:
    removed_indexes = original_pjm_indexes.difference(pjm.index)

    # Remove only the discarded PJM rows from the original df.
    df = df.drop(index=removed_indexes)

    # Update only MW values for the retained rows.
    df.loc[pjm.index, "MW"] = pjm["MW"]

    duplicate_report = count_duplicate_rows(df)
    print(duplicate_report)
    assert (duplicate_report == 0)

    new_length = len(df)
    print(f"Length comparison {original_length} != {new_length}: {original_length != new_length}")
    print("Saving....")
    df.to_csv("../data/cleaned/all_data_relabelled_CLEANED.csv")
    print("Done Saving")

def checking():
    df = pd.read_csv("../data/all_data_relabelled.csv")
    # Columns
    DATETIME = "Datetime"
    MEGAWATTS = "MW"
    COMPANY = "Company"

    PJM = "PJM"

    pjm = df.loc[df[COMPANY].eq(PJM)].copy()
    duplicate_count, pjm_cleaned = check_duplicates(pjm)

    pjm.drop_duplicates(inplace=True)
    print(duplicate_count)
    print(len(pjm))
    print(len(pjm_cleaned))
    #pjm.reset_index(drop=True, inplace=True)
    print(pjm.index)



