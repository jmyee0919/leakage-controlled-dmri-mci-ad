"""Check the public snapshot without any participant-level inputs."""
from pathlib import Path
import csv, hashlib, json, math

ROOT = Path(__file__).resolve().parents[1]

def main():
    inventory = json.loads((ROOT/'MANIFEST.json').read_text())
    for item in inventory['files']:
        p = ROOT/item['path']
        if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise AssertionError('File changed: '+item['path'])
    tables = list((ROOT/'data/aggregate').glob('*.csv'))
    assert len(tables) == 32
    for p in ROOT.rglob('*.csv'):
        with p.open(newline='') as f: rows = list(csv.reader(f))
        assert rows and all(len(r) == len(rows[0]) for r in rows), p
    with (ROOT/'data/aggregate/Table_S21_sensitivity_added_in_revision.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    anchors = {'Primary (reference)':.0793650793650793,
               'Excluding AD <=30 days':.0773809523809523,
               'Excluding AD <=45 days':.0884353741496598,
               'Field strength added':.0634920634920635}
    for label, expected in anchors.items():
        row = next(r for r in rows if r['analysis'] == label)
        assert math.isclose(float(row['dAUC']), expected, abs_tol=1e-8),label
    figures = list((ROOT/'figures').rglob('*.pdf'))
    assert len(figures) == 6
    for p in ROOT.rglob('*.py'): compile(p.read_text(), str(p), 'exec')
    print('PASS: public file integrity, 32 aggregate CSVs, six figure sets, S21 anchors and Python syntax.')
    print('Scope: aggregate snapshot consistency; no raw clinical adjudication or external validation.')

if __name__ == '__main__': main()
