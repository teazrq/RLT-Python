"""Pack CSVs from generate_survival_smoother.R into reviewed test fixtures.

Usage: python pack_survival_smoother.py <CSV-directory>
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

source = Path(sys.argv[1])
target = Path(__file__).parent
metadata = list(csv.DictReader((source / 'metadata.csv').open()))
arrays = {}
for record in metadata:
    tag = record['id']
    for field in ('raw', 'smooth', 'band_sd'):
        path = source / f'{tag}_{field}.csv'
        if path.exists():
            if field == 'raw':
                n = int(record['n'])
                arrays[f'{tag}_{field}'] = np.fromfile(source / f'{tag}_raw.bin', dtype='<f8').reshape(n, n, order='F')
            else:
                arrays[f'{tag}_{field}'] = np.loadtxt(path, delimiter=',', skiprows=1)
for name in ('design', 'penalty1', 'penalty2', 'fixed_sp_surface',
             'mc_draws', 'mc_widths', 'mc_scales'):
    arrays[name] = np.loadtxt(source / f'{name}.csv', delimiter=',', skiprows=1)
np.savez_compressed(target / 'survival_smoother_mgcv_1_9_4.npz', **arrays)
manifest = {'versions': (source / 'versions.txt').read_text().splitlines(),
            'reference': 'corrected local RLT source and mgcv 1.9-4',
            'cases': metadata}
(target / 'survival_smoother_mgcv_1_9_4.json').write_text(json.dumps(manifest, indent=2) + '\n')
