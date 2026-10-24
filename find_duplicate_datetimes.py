"""Find repeated timestamps in data/cleaned and save data/duplicate_datetimes.csv.

Check Datetime only, not MW. Include every row in each duplicate group so the
measurements can be compared. The input files are left unchanged.
"""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / 'data' / 'cleaned'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'duplicate_datetimes.csv'


def find_duplicate_datetimes(input_dir=DEFAULT_INPUT_DIR, output_path=DEFAULT_OUTPUT):
    """Save rows with repeated timestamps and return them as a DataFrame."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    paths = sorted(input_dir.glob('*.csv'))
    if not paths:
        raise ValueError(f'No CSV files found in {input_dir}')
    if output_path.resolve().parent == input_dir.resolve():
        raise ValueError('Save the report outside the input directory to avoid scanning it on later runs.')
    for path in paths:
        if path.resolve() == output_path.resolve() or (
            output_path.exists() and path.samefile(output_path)
        ):
            raise ValueError('The output must not overwrite a source CSV.')

    frames = []
    total_extra = 0
    total_groups = 0
    print('Checking for repeated Datetime values.\n')
    for path in paths:
        data = pd.read_csv(path)
        if 'Datetime' not in data.columns:
            raise ValueError(f'{path.name}: missing Datetime column.')
        value_columns = [column for column in data.columns if column.endswith('_MW')]
        if len(value_columns) != 1:
            raise ValueError(f'{path.name}: expected exactly one measurement column ending in _MW.')
        timestamps = pd.to_datetime(data['Datetime'], errors='raise')
        if timestamps.isna().any():
            raise ValueError(f'{path.name}: missing Datetime values.')

        mask = timestamps.duplicated(keep=False)
        extra = int(timestamps.duplicated(keep='first').sum())
        groups = int(timestamps.loc[mask].nunique())
        total_extra += extra
        total_groups += groups
        print(f'{path.name}: {len(data):,} rows; {extra:,} extra duplicate occurrences; '
              f'{int(mask.sum()):,} rows in duplicate groups')

        column = value_columns[0]
        duplicates = data.loc[mask, ['Datetime', column]].rename(columns={column: 'MW'})
        duplicates.insert(0, 'Company', path.stem.removesuffix('_hourly_CLEANED'))
        duplicates['SourceFile'] = path.name
        frames.append(duplicates)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(['Company', 'Datetime'], kind='stable').reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f'\nTotal extra duplicate occurrences: {total_extra:,}')
    print(f'Total duplicate timestamp groups (within files): {total_groups:,}')
    print(f'Total rows exported, including first occurrences: {len(result):,}')
    print(f'Output: {output_path}')
    print('Input files unchanged.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        find_duplicate_datetimes(args.input_dir, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
