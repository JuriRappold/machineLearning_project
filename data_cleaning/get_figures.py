import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
from plot_outliers import calculate_outliers, plot_outliers

cwd = Path.cwd().resolve()
project_root = next(
    (path for path in (cwd, *cwd.parents) if (path / 'plot_outliers.py').is_file()),
    None,
)
if project_root is None:
    raise FileNotFoundError('Run this notebook from the project root or notebooks directory.')

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

cleaned_dir = project_root / 'data' / 'cleaned'
threshold = 3.0

def plot_cleaned_file(filename):
    """Plot the load values from one cleaned CSV."""
    data = pd.read_csv(cleaned_dir / filename)
    company = filename.removesuffix("_hourly_CLEANED.csv")
    value_columns = [column for column in data.columns if column.endswith('MW')]
    if len(value_columns) != 1:
        raise ValueError(f'{filename}: expected exactly one load column ending in MW.')
    value_column = value_columns[0]
    data = data[['Datetime', value_column]].copy()  # Ignore saved CSV index columns.
    result, fig, ax = plot_outliers(
        data, value_column, date_column='Datetime', threshold=threshold,
        title=f'{company} cleaned hourly load: Z-score outliers',
    )
    mean = result[value_column].mean()
    std = result[value_column].std(ddof=1)
    print(filename)
    print(f'Mean: {mean:,.2f} MW; sample standard deviation: {std:,.2f} MW')
    print(f'Mean ±{threshold:g} SD: [{mean - threshold * std:,.2f}, '
          f'{mean + threshold * std:,.2f}] MW')
    print(f"Outliers: {result['outlier'].sum():,} / {len(result):,} "
          f"({result['outlier'].mean():.2%})")
    #plt.show()
    #plt.close(fig)score_outliers
    return result


def get_all_cleaned_figures():
    plot_cleaned_file('AEP_hourly_CLEANED.csv')
    plot_cleaned_file('COMED_hourly_CLEANED.csv')
    plot_cleaned_file('DAYTON_hourly_CLEANED.csv')
    plot_cleaned_file('DEOK_hourly_CLEANED.csv')
    plot_cleaned_file('DOM_hourly_CLEANED.csv')
    plot_cleaned_file('DUQ_hourly_CLEANED.csv')
    plot_cleaned_file('EKPC_hourly_CLEANED.csv')
    plot_cleaned_file('FE_hourly_CLEANED.csv')
    plot_cleaned_file('NI_hourly_CLEANED.csv')
    plot_cleaned_file('PJME_hourly_CLEANED.csv')
    plot_cleaned_file('PJM_Load_hourly_CLEANED.csv')
    plot_cleaned_file('PJMW_hourly_CLEANED.csv')
    plot_cleaned_file("all_data_relabelled_CLEANED.csv")

if __name__ =="__main__":
    get_all_cleaned_figures()