"""Remove rows outside the combined dataset's mean +/- 3 SD.

Read data/all_data.csv and save data/all_data_outliers_removed.csv.
Use sample SD (ddof=1), calculated once from all companies together.
"""

import argparse
from pathlib import Path

import pandas as pd

from plot_outliers import calculate_outliers


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / 'data' / 'cleaned' /'all_data_relabelled_CLEANED.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'all_data_READY.csv'


def remove_outliers(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, threshold=3.0):
    """Save rows with absolute Z-scores at or below the threshold."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve() or (
        output_path.exists() and input_path.samefile(output_path)
    ):
        raise ValueError('Input and output must be different files.')
    if output_path.resolve() == DEFAULT_INPUT.resolve() or (
        output_path.exists() and DEFAULT_INPUT.exists() and output_path.samefile(DEFAULT_INPUT)
    ):
        raise ValueError('The output must not overwrite data/all_data.csv.')

    data = pd.read_csv(input_path)
    if 'MW' not in data.columns:
        raise ValueError('The input CSV must contain an MW column.')
    scored = calculate_outliers(data, 'MW', threshold)
    mean = scored['MW'].mean()
    std = scored['MW'].std(ddof=1)
    lower, upper = mean - threshold * std, mean + threshold * std
    # Keep the original columns; Z-scores and flags are only used for filtering.
    result = data.loc[~scored['outlier']].copy()
    removed = int(scored['outlier'].sum())
    if len(result) + removed != len(data):
        raise RuntimeError('Filtered row counts do not match the input.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f'Input: {input_path}')
    print(f'Mean: {mean:,.2f} MW; sample SD: {std:,.2f} MW')
    print(f'Mean +/- {threshold:g} SD: [{lower:,.2f}, {upper:,.2f}] MW')
    print(f'Rows before: {len(data):,}')
    print(f'Outlier rows removed: {removed:,}')
    print(f'Rows after: {len(result):,}')
    print(f'Output: {output_path}')
    return result


def remove_outliers_py():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--threshold', type=float, default=3.0)
    args = parser.parse_args()
    try:
        remove_outliers(args.input, args.output, args.threshold)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    remove_outliers_py()
