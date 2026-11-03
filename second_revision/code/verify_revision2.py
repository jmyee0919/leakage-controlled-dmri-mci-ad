"""Check public second-revision arithmetic and archived file integrity."""
from pathlib import Path
import csv,json,statistics,math

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'second_revision/Data_file_S1'
def read(n):
    with (D/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    checks=0
    for raw,summary in [('Table_S11_four_model_each_repeat.csv','Table_2_3_four_model_repeatedCV_summary.csv'),('Table_S13_broader_each_repeat.csv','Table_S13_broader_four_model_repeatedCV.csv')]:
        rr=read(raw)
        for row in rr:
            for x,z in [('M2','M1'),('M3','M1'),('M4','M3'),('M4','M2')]:
                k=f'dAUC_{x}-{z}'
                if k not in row:continue
                assert abs(float(row[k])-(float(row['AUC_'+x])-float(row['AUC_'+z])))<=1.51e-6
                checks+=1
        for row in read(summary):
            vals=[float(r[row['quantity']]) for r in rr if r['analysis']==row['analysis']]
            for k,v in [('mean',statistics.mean(vals)),('sd',statistics.stdev(vals)),('min',min(vals)),('max',max(vals))]:
                assert abs(float(row[k])-v)<=1.1e-6;checks+=1
            if row['negative_repeats']:
                assert int(float(row['negative_repeats']))==sum(v<0 for v in vals);checks+=1
    for f in D.glob('*.csv'):
        with f.open(encoding='utf-8-sig',newline='') as h:r=list(csv.reader(h))
        assert r and all(len(x)==len(r[0]) for x in r);checks+=1
    assert 'Estimated intracranial volume, cm³' not in (D/'Table_S02_classified_vs_unclassified.csv').read_text()
    assert len(read('Table_S28B_training_vs_holdout.csv'))==17
    report=json.loads((ROOT/'second_revision/verification/REPRODUCTION_REPORT.json').read_text())
    assert report['status']=='PASS' and report['checks']==690
    assert report['maximum_manual_prediction_difference']<=5.1e-14
    for p in (ROOT/'second_revision/code').glob('*.py'):compile(p.read_text(),str(p),'exec')
    print(f'PASS: {checks} public aggregate checks; CSV structure, unit label, partition table and reproduction record.')
if __name__=='__main__':main()
