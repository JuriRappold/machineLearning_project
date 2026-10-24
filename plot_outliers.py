"""Generic Z-score outlier detection and plotting, without file-writing side effects."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_outliers(data, column, threshold=3.0):
    """Return a copy with Z_score and outlier columns, using sample std (ddof=1)."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('The Z-score threshold must be finite and positive.')
    result = data.copy()
    values = pd.to_numeric(result[column], errors='raise')
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite, nonmissing measurements are required.')

    # Same calculation as outliers.py, parameterized by the measurement column.
    mean = values.mean()
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        raise ValueError('Z-scores require a finite, nonzero standard deviation.')
    result[column] = values
    result['Z_score'] = (values - mean) / std
    result['outlier'] = result['Z_score'].abs() > threshold
    return result


def plot_outliers(data, column, *, date_column=None, threshold=3.0, title=None):
    """Return (flagged data, figure, axes). Do not show or save the chart."""
    result = calculate_outliers(data, column, threshold)
    if date_column is not None:
        if result[date_column].isna().any():
            raise ValueError('Dates must not be missing.')
        result[date_column] = pd.to_datetime(result[date_column], errors='raise')
        if result[date_column].isna().any():
            raise ValueError('Dates must not be missing.')
        result = result.sort_values(date_column)

    mean = result[column].mean()
    std = result[column].std(ddof=1)
    lower, upper = mean - threshold * std, mean + threshold * std
    fig, ax = plt.subplots(figsize=(15, 6))
    for flag, color, label, size, alpha in [
        (False, '#2463a6', f'Within ±{threshold:g} SD', 5, 0.35),
        (True, '#d44335', f'Outliers (|Z-score| > {threshold:g})', 24, 0.9),
    ]:
        subset = result.loc[result['outlier'] == flag]
        x = subset[date_column] if date_column is not None else subset.index
        ax.scatter(x, subset[column], color=color, label=label, s=size, alpha=alpha,
                   linewidths=0, zorder=3 if flag else 1, rasterized=True)

    ax.axhline(mean, color='#333333', linewidth=1.3, label=f'Mean: {mean:,.2f}')
    ax.axhline(upper, color='#bc6800', linestyle='--', linewidth=1.5,
               label=f'Mean + {threshold:g} SD: {upper:,.2f}')
    ax.axhline(lower, color='#bc6800', linestyle='--', linewidth=1.5,
               label=f'Mean − {threshold:g} SD: {lower:,.2f}')
    ax.set(title=title or f'{column}: Z-score outliers',
           xlabel=date_column if date_column is not None else 'Observation index',
           ylabel=column)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc='upper left', fontsize=9)
    if date_column is not None:
        fig.autofmt_xdate()
    fig.tight_layout()
    title = title.replace(" ", "_")
    fig.savefig(fname=f"./figures/{title}.png")
    return result, fig, ax


def plot():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='CSV file containing measurements')
    parser.add_argument('--column', required=True, help='Measurement column to analyze')
    parser.add_argument('--date-column', help='Optional timestamp column (otherwise use row index)')
    parser.add_argument('--threshold', type=float, default=3.0, help='Z-score cutoff (default: 3)')
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    result, fig, _ = plot_outliers(
        data, args.column, date_column=args.date_column, threshold=args.threshold,
    )
    print(f"Outliers: {result['outlier'].sum():,} / {len(result):,} "
          f"({result['outlier'].mean():.2%})")
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    plot()
