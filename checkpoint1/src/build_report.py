"""Build exactly five readable pages from editable Markdown and verified run output."""
from pathlib import Path
import json, re, html
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.graphics.shapes import Drawing, Line, String, Circle
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/ColdCast_Checkpoint_I.pdf'
INK=colors.HexColor('#17304b'); TEAL=colors.HexColor('#087d83')

def markup(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    return s

def chart(data):
    d=Drawing(480,115)
    for offset,name,label,col in [(0,'fusion_mlp','Fusion MLP',INK),(245,'fusion_temperature','Temperature calibrated',TEAL)]:
        x0,y0,w,h=offset+34,22,160,72
        d.add(String(x0,103,label,fontName='Helvetica-Bold',fontSize=8,fillColor=col))
        d.add(Line(x0,y0,x0+w,y0+h,strokeColor=colors.lightgrey,strokeDashArray=[2,2]))
        d.add(Line(x0,y0,x0+w,y0,strokeColor=INK));d.add(Line(x0,y0,x0,y0+h,strokeColor=INK))
        points=data['metrics'][name]['reliability']
        for r in points:
            d.add(Circle(x0+w*r['confidence'],y0+h*r['accuracy'],2.2,fillColor=col,strokeColor=col))
        for v in [0,.5,1]:
            d.add(String(x0+w*v-3,y0-10,str(v),fontSize=6))
            d.add(String(x0-16,y0+h*v-2,str(v),fontSize=6))
        d.add(String(x0+42,1,'Predicted probability',fontSize=7))
        d.add(String(x0+4,y0+h-10,'Observed correctness',fontSize=6,fillColor=INK))
    return d

def main():
    data=json.loads((ROOT/'results/synthetic/metrics.json').read_text())
    if not data['provenance'].startswith('SYNTHETIC'):
        raise ValueError('This report explicitly describes synthetic evidence')
    rows=['| Method | Brier | Log loss | ECE |','| --- | --- | --- | --- |']
    names={'constant_0.5':'Constant 0.5','global_rate':'Global rate','system_rate':'Smoothed system rate',
           'text_logistic':'Hashed text logistic','metadata_logistic':'Metadata logistic','fusion_mlp':'Fusion MLP',
           'fusion_temperature':'Fusion + temperature'}
    for key,name in names.items():
        m=data['metrics'][key]
        rows.append(f"| {name} | {m['brier']:.4f} | {m['log_loss']:.4f} | {m['ece']:.4f} |")
    source=(ROOT/'report_source.md').read_text().replace('{{RESULTS_TABLE}}',
        '**Table 1. Synthetic software-check results; not competition performance.**\n\n'+'\n'.join(rows))
    (ROOT/'output/report.md').write_text(source.replace('{{RELIABILITY_FIGURE}}',
        'Figure 1. Synthetic reliability diagrams appear in the PDF. Ten fixed-width bins; empty bins omitted.'))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body',fontName='Times-Roman',fontSize=10.1,leading=12.5,spaceAfter=6,textColor=INK))
    styles.add(ParagraphStyle(name='SmallRef',parent=styles['Body'],fontSize=8.4,leading=10.2,spaceAfter=4))
    styles.add(ParagraphStyle(name='H1x',fontName='Helvetica-Bold',fontSize=16,leading=18.5,spaceAfter=9,textColor=INK))
    styles.add(ParagraphStyle(name='H2x',fontName='Helvetica-Bold',fontSize=12,leading=14,spaceBefore=8,spaceAfter=7,textColor=TEAL))
    styles.add(ParagraphStyle(name='H3x',fontName='Helvetica-Bold',fontSize=10.2,leading=12,spaceBefore=6,spaceAfter=5,textColor=INK))
    styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=8.2,leading=10.5,textColor=INK))
    def table(lines):
        cells=[[c.strip() for c in line.strip('|').split('|')] for line in lines if not re.match(r'^\|\s*---',line)]
        count=len(cells[0]); widths={2:[150,342],3:[126,166,200],4:[240,84,84,84]}[count]
        tab=Table([[Paragraph(markup(c),styles['Cell']) for c in row] for row in cells],colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1eff0')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('LINEBELOW',(0,0),(-1,0),.6,TEAL),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe5'))]))
        return tab
    story=[]
    for page_index,page in enumerate(source.split('<!-- PAGEBREAK -->')):
        if page_index: story.append(PageBreak())
        blocks=re.sub(r'(?m)^(#{1,3} [^\n]+)\n', r'\1\n\n', page.strip()).split('\n\n')
        refs=False
        for block in blocks:
            if block.startswith('# '): story.append(Paragraph(markup(block[2:]),styles['H1x']))
            elif block.startswith('## '):
                refs=block.strip()=='## References'
                story.append(Paragraph(markup(block[3:]),styles['H2x']))
            elif block.startswith('### '): story.append(Paragraph(markup(block[4:]),styles['H3x']))
            elif block.startswith('|'): story.extend([table(block.splitlines()),Spacer(1,7)])
            elif block.strip()=='{{RELIABILITY_FIGURE}}':
                story.append(KeepTogether([chart(data),Paragraph('Figure 1. Synthetic reliability diagrams; dashed line = ideal agreement. Empty bins omitted.',styles['SmallRef'])]))
            else: story.append(Paragraph(markup(block.replace('\n',' ')),styles['SmallRef' if refs else 'Body']))
    def footer(canvas,doc):
        canvas.setStrokeColor(TEAL);canvas.line(54,751,558,751)
        canvas.setFont('Helvetica',8);canvas.setFillColor(INK)
        canvas.drawString(54,761,'COLDCAST / PROJECT CHECKPOINT I')
        canvas.drawRightString(558,761,'19 OCTOBER 2026 / TEAM REVIEW DRAFT')
        canvas.line(54,42,558,42)
        canvas.drawString(54,29,'CS5814 - Virginia Tech | Synthetic verification clearly separated from research results')
        canvas.drawRightString(558,29,str(doc.page))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUT),pagesize=(612,792),rightMargin=60,leftMargin=60,topMargin=52,bottomMargin=53,
        title='ColdCast - Project Checkpoint I',author='ColdCast team; assistant-supported draft')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(OUT)
    check=dict(pages=len(reader.pages),words_by_page=[len(p.extract_text().split()) for p in reader.pages],
        synthetic_rows=sum(v['rows'] for v in data['counts'].values()),status='DRAFT: human contribution review and live publication pending')
    (ROOT/'output/build_check.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check,indent=2))
    if len(reader.pages)!=5: raise RuntimeError('Layout must be repaired to exactly 5 pages')

if __name__=='__main__': main()
