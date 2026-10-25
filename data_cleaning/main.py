from clean import clean
from split_and_merge import merge
from relabel_companies import relabel
from average_duplicates_after_joining_of_PJM import avg_after_relabel
from remove_outliers import remove_outliers_py
from get_figures import get_all_cleaned_figures

def data_preparation():
    clean()
    merge()
    relabel()
    avg_after_relabel()
    get_all_cleaned_figures()
    remove_outliers_py()

if __name__ == "__main__":
    data_preparation()