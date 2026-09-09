"""Summarize arena_profile --lifetimes CSV without losing the exact-age source."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

source = Path(sys.argv[1])
histograms = defaultdict(dict)
with source.open() as file:
    for row in csv.DictReader(file):
        key = (int(row['step']), row['status'])
        histograms[key][int(row['sweeps_survived'])] = int(row['node_count'])

bounds = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31),
          (32, 63), (64, 127), (128, 255), (256, None)]
with source.with_suffix('.buckets.csv').open('w') as file:
    writer = csv.writer(file)
    writer.writerow(['step', 'status', 'age_min', 'age_max', 'node_count'])
    for (step, status), counts in sorted(histograms.items()):
        for low, high in bounds:
            writer.writerow([step, status, low, high if high is not None else '',
                             sum(n for age, n in counts.items()
                                 if age >= low and (high is None or age <= high))])

with source.with_suffix('.summary.csv').open('w') as file:
    writer = csv.writer(file)
    writer.writerow(['step', 'status', 'nodes', 'mean_age', 'p50', 'p90', 'p99',
                     'max_age', 'age_ge_16', 'age_ge_32', 'age_ge_64', 'age_ge_128'])
    for (step, status), counts in sorted(histograms.items()):
        total = sum(counts.values())
        def quantile(percent):
            cumulative = 0
            for age, n in sorted(counts.items()):
                cumulative += n
                if total and cumulative * 100 >= total * percent:
                    return age
            return ''
        writer.writerow([step, status, total,
                         sum(age * n for age, n in counts.items()) / total if total else '',
                         quantile(50), quantile(90), quantile(99),
                         max((age for age, n in counts.items() if n), default=''),
                         *(sum(n for age, n in counts.items() if age >= cutoff)
                           for cutoff in [16, 32, 64, 128])])
