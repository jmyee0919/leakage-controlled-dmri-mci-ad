"""Preserve the archived vector figure and clarify its last-known MCI labels."""
from pathlib import Path
import argparse
import fitz
from PIL import Image
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);d=fitz.open(a.source);p=d[0]
    matches=[]
    for block in p.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for s in line['spans']:
                if s['text']=='Recorded MCI':matches.append(s)
    if len(matches)!=2:raise ValueError('Expected two archived labels')
    # Intersect the target glyphs without touching the neighboring prediction-time line.
    for s in matches:
        r=fitz.Rect(s['bbox']);r.y1=r.y0+4.8
        p.add_redact_annot(r,fill=(1,1,1))
    p.apply_redactions(images=0,graphics=0)
    font=fitz.Font('tiro');txt='Last-known recorded MCI';size=7.8
    for s in matches:
        x=(s['bbox'][0]+s['bbox'][2])/2-font.text_length(txt,fontsize=size)/2
        p.insert_text((x,s['origin'][1]),txt,fontname='tiro',fontsize=size,color=(.29,.29,.29))
    d.save(a.out/'Figure_1.pdf',garbage=4,deflate=True)
    # Reopen the written PDF so raster dimensions follow its saved page box.
    reopened=fitz.open(a.out/'Figure_1.pdf')
    pix=reopened[0].get_pixmap(dpi=600,alpha=False)
    Image.frombytes('RGB',[pix.width,pix.height],pix.samples).save(a.out/'Figure_1.tif',compression='tiff_lzw',dpi=(600,600))
if __name__=='__main__':main()
