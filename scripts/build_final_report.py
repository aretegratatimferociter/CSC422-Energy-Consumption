"""Build the editable final report from the reviewed Markdown source."""

import re
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def main():
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(0.7)
    section.left_margin = section.right_margin = Inches(0.8)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2']:
        style = document.styles[name]
        style.font.name = 'Times New Roman'
        fonts = style.element.get_or_add_rPr().find(qn('w:rFonts'))
        if fonts is not None:
            for attribute in list(fonts.attrib):
                if 'Theme' in attribute:
                    del fonts.attrib[attribute]
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.size = Pt(11 if name == 'Normal' else 15)
    document.styles['Normal'].paragraph_format.space_after = Pt(6)
    document.styles['Normal'].paragraph_format.line_spacing = 1.08
    header = section.header.paragraphs[0]
    header.text = 'CSC 422 · Fall 2026                                  Final Project Report'
    header.runs[0].font.size = Pt(9)
    footer = section.footer.paragraphs[0]
    footer.alignment = 1
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)

    lines = (ROOT / 'reports/final_report.md').read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                values = [value.strip() for value in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', v) for v in values):
                    rows.append(values)
                i += 1
            table = document.add_table(rows=0, cols=len(rows[0]))
            table.style = 'Table Grid'
            for r, values in enumerate(rows):
                cells = table.add_row().cells
                for cell, value in zip(cells, values):
                    cell.text = value
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(9)
                        run.bold = r == 0
                    if r == 0:
                        shade = OxmlElement('w:shd')
                        shade.set(qn('w:fill'), 'EEEEEE')
                        cell._tc.get_or_add_tcPr().append(shade)
                cant = OxmlElement('w:cantSplit')
                table.rows[-1]._tr.get_or_add_trPr().append(cant)
            repeat = OxmlElement('w:tblHeader')
            table.rows[0]._tr.get_or_add_trPr().append(repeat)
            document.add_paragraph()
            continue
        if line.startswith('# '):
            paragraph = document.add_paragraph(line[2:], 'Title')
            paragraph.alignment = 1
            paragraph.runs[0].font.size = Pt(22)
            borders = paragraph._p.get_or_add_pPr().find(qn('w:pBdr'))
            if borders is not None:
                paragraph._p.pPr.remove(borders)
        elif line.startswith('## '):
            document.add_heading(line[3:], level=1)
        else:
            paragraph = document.add_paragraph(line.replace('`', ''))
            paragraph.alignment = 0 if 'https://' in line else 3
        i += 1
    document.core_properties.title = 'Time Series Forecasting of Energy Consumption'
    document.core_properties.author = 'David Valenta, Cole Dail, Peter Holt, Thomas Mitchell'
    document.save(ROOT / 'reports/CSC422_Final_Report.docx')


if __name__ == '__main__':
    main()
