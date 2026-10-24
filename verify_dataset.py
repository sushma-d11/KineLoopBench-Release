#!/usr/bin/env python3
"""Read-only validation of the KineLoopBench GitHub review draft (standard library)."""
import csv,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read_csv(path):
 with (ROOT/path).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

hashes=read_csv('subsets/file_sha256.csv')
for r in hashes:
 actual=hashlib.sha256((ROOT/r['relative_path']).read_bytes()).hexdigest()
 assert actual==r['sha256'],r['relative_path']
for filename,n in [('primary_225.csv',225),('feasible_150.csv',150),('matched_198.csv',198),('historical_75_excluded.csv',75)]:
 assert len(read_csv('subsets/'+filename))==n,filename
primary={r['case_uid'] for r in read_csv('subsets/primary_225.csv')}
feasible={r['case_uid'] for r in read_csv('subsets/feasible_150.csv')}
matched={r['case_uid'] for r in read_csv('subsets/matched_198.csv')}
historical={r['case_uid'] for r in read_csv('subsets/historical_75_excluded.csv')}
assert feasible <= primary and matched <= primary and not (historical & primary)
assert len(primary | historical)==300
expected='911bf9597322356f2af1d1df7917b1dd25b38d81bff7db68baf6e0a44f77f554'
assert hashlib.sha256((ROOT/'code/scoring_core_frozen_v1.py').read_bytes()).hexdigest()==expected
for fam in ('RRRR','RRRP','5R'):
 canon=(ROOT/'assets/model_input_schematics'/f'{fam}.png').read_bytes()
 for r in read_csv('subsets/primary_225.csv'):
  if r['mechanism_family']!=fam: continue
  p=ROOT/'data/primary_225'/fam/r['sample_id']/'image.png'
  assert p.read_bytes()==canon,(fam,r['case_uid'])
print('PASS: 300 immutable original cases preserved; 225 primary, 150 feasible, 198 matched, 75 historical; frozen scorer and schematics checked.')
print('Note: model outputs are NOT bundled in this public review draft; full scoring reproduction needs approved archived responses.')
