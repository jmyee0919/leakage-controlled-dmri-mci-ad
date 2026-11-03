"""Clarify that both numerical plausibility thresholds apply to absolute values."""
import argparse
from pathlib import Path
import fitz
from PIL import Image
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);d=fitz.open(a.source);p=d[0];spans=[]
    for b in p.get_text('dict')['blocks']:
        for line in b.get('lines',[]):
            for s in line['spans']:
                if 'Rule A,' in s['text'] or 'Rule B,' in s['text']:spans.append(s)
    if len(spans)!=2:raise ValueError('Expected two archived threshold labels')
    for s in spans:p.add_redact_annot(fitz.Rect(s['bbox']),fill=(1,1,1))
    p.apply_redactions(images=0,graphics=0)
    for s in spans:
        text='   Rule A, absolute ICV change > 5% (n = 11)' if 'Rule A' in s['text'] else '   Rule B, |regional robust z| > 4 (n = 4)'
        p.insert_text(s['origin'],text,fontname='tiro',fontsize=7.6,color=(.067,.067,.067))
    d.save(a.out/'Figure_2.pdf',garbage=4,deflate=True)
    reopened=fitz.open(a.out/'Figure_2.pdf')
    pix=reopened[0].get_pixmap(dpi=600,alpha=False)
    Image.frombytes('RGB',[pix.width,pix.height],pix.samples).save(a.out/'Figure_2.tif',compression='tiff_lzw',dpi=(600,600))
if __name__=='__main__':main()
