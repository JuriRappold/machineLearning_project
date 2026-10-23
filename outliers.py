import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
"""
threshold = 3.0
mean = df['Data'].mean()
std = df['Data'].std(ddof=1)
if not np.isfinite(std) or std == 0:
    raise ValueError('Z-scores require a finite, nonzero standard deviation.')
df['Z_score'] = (df['Data'] - mean) / std
df['outlier'] = df['Z_score'].abs() > threshold
lower, upper = mean - threshold * std, mean + threshold * std
print(f'Mean = {mean:.3f}; sample standard deviation = {std:.3f}')
print(f'Unflagged interval: [{lower:.3f}, {upper:.3f}]')
display(df.loc[df['outlier']].round(3))
print(f"Flagged: {df['outlier'].sum()} / {len(df)}")
"""

aep = pd.read_csv("data/original/AEP_hourly.csv")

threshold = 1.5#1.0#2.0#2.5#3.0
plt.rcParams.update({'figure.figsize': (9, 4), 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, RED = '#2463a6', '#d44335'
#matplotlib.use('QtAgg')
# MW /h
mean=aep["AEP_MW"].mean()
std=aep["AEP_MW"].std(ddof=1)
if not np.isfinite(std) or std == 0:
    raise ValueError('Z-scores require a finite, nonzero standard deviation.')
aep["MW_z-score"] = (aep["AEP_MW"]-mean) / std
aep["outlier"] = aep["MW_z-score"].abs() > threshold
lower, upper = mean - threshold * std, mean + threshold * std
print(f'Mean = {mean:.3f}; sample standard deviation = {std:.3f}')
print(f'Unflagged interval: [{lower:.3f}, {upper:.3f}]')
display(aep.loc[aep['outlier']].round(3))
print(f"Flagged: {aep['outlier'].sum()} / {len(aep)}")
print(f"Percentage: {(aep['outlier'].sum()/float(len(aep)))*100}%")
aep.to_csv(f"./data/AEP_hourly_with_outliers{threshold}.csv")


fig, axes = plt.subplots(2, 1, figsize=(10, 7))
for ax, column, limits, title, ylabel in [
    (axes[0], 'AEP_MW', (lower, upper), 'Measurements and detection boundaries', 'Measurement'),
    (axes[1], 'MW_z-score', (-threshold, threshold), 'Z-scores and detection boundaries', 'Z-score')
]:
    for flag, color, label in [(False, BLUE, 'Not flagged'), (True, RED, 'Flagged')]:
        subset = aep.loc[aep['outlier'] == flag]
        ax.scatter(subset.index, subset[column], color=color, label=label, s=28)
    for boundary in limits:
        ax.axhline(boundary, color='gray', linestyle='--')
    ax.set(title=title, xlabel='Observation index', ylabel=ylabel)
    ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"./figures/outliers_AEP{threshold}.png")
#plt.show()