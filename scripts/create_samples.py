"""Rebuild the synthetic sample library: python scripts/create_samples.py."""
from pathlib import Path
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

OUTPUT = Path(__file__).resolve().parents[1] / 'samples'
OUTPUT.mkdir(exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('DemoTitle', fontName='Helvetica-Bold', fontSize=25, leading=31,
                          textColor=HexColor('#163f42'), spaceAfter=18))
styles.add(ParagraphStyle('DemoBody', fontName='Helvetica', fontSize=11, leading=17, spaceAfter=12))
styles.add(ParagraphStyle('DemoHeading', fontName='Helvetica-Bold', fontSize=14, leading=20,
                          textColor=HexColor('#163f42'), spaceBefore=12, spaceAfter=8))


def footer(canvas, doc):
    canvas.setStrokeColor(HexColor('#d2dddd'))
    canvas.line(48, 48, A4[0]-48, 48)
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(HexColor('#527071'))
    canvas.drawString(48, 32, 'Synthetic sample | Nitesh Kelwani - Document Q&A')
    canvas.drawRightString(A4[0]-48, 32, f'Page {doc.page}')


def build(filename, title, pages):
    story = []
    for index, sections in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(title if index == 0 else title + ' - continued', styles['DemoTitle']))
        if index == 0:
            story.append(Paragraph('Fictional company and policies. Created solely for this portfolio demo.', styles['DemoBody']))
        for heading, body in sections:
            story.append(Paragraph(heading, styles['DemoHeading']))
            story.append(Paragraph(body, styles['DemoBody']))
    SimpleDocTemplate(str(OUTPUT / filename), pagesize=A4, rightMargin=48, leftMargin=48,
                      topMargin=52, bottomMargin=66, title=title, author='Nitesh Kelwani').build(
                          story, onFirstPage=footer, onLaterPages=footer)


build('company-handbook.pdf', 'Northstar Company Handbook', [
    [('Remote work', 'Employees may work remotely up to three days per week. Team members must be available '
      'between 10:00 and 16:00 India Standard Time on working days. Managers approve the weekly schedule.'),
     ('Annual leave', 'Full-time employees receive 20 days of paid annual leave per calendar year. '
      'Submit planned leave requests at least five working days in advance. Up to five unused days may carry into the next year.'),
     ('Equipment', 'The company provides one laptop and a headset. Report broken equipment to the internal IT desk. '
      'Personal devices must not store company documents.')],
    [('Expense reimbursement', 'Submit expense receipts within 14 calendar days of the purchase. '
      'The finance team processes approved claims within seven working days. Manager approval is required for purchases above INR 5,000.'),
     ('Learning allowance', 'Each employee has an annual learning allowance of INR 15,000 for relevant courses and books. '
      'Obtain manager approval before purchasing a course. Unused learning allowance does not carry forward.'),
     ('Support', 'For policy questions, contact the fictional People Operations team at people@northstar.example. '
      'This sample does not define salaries, medical benefits or disciplinary procedures.')]
])

build('product-guide.pdf', 'Atlas Product Guide', [
    [('Create a workspace', 'Atlas is a fictional analytics product. To create a workspace, open Settings, choose Workspaces, '
      'then select Create workspace. Enter a name and invite team members. A workspace has one owner and up to ten editors.'),
     ('Import data', 'Open Data Sources and select Upload CSV. Files must be UTF-8 encoded, no larger than 10 MB, '
      'and contain a header row. Atlas supports up to 50,000 rows per imported file.'),
     ('Build a report', 'Open Reports and choose New report. Select a data source, choose dimensions and metrics, '
      'and select Run report. Save the report with a descriptive title.')],
    [('Export a report', 'Open a saved report, select Export in the upper-right corner, and choose CSV or PDF. '
      'CSV exports include the filtered rows. PDF exports include the visible charts and summary. Only owners and editors can export reports.'),
     ('Access roles', 'Owners manage billing, workspace settings and invitations. Editors can import data, create reports and export results. '
      'Viewers can read reports but cannot edit, import or export.'),
     ('Troubleshooting', 'If an import fails, check the file size, encoding and header row. If a report appears empty, '
      'clear date filters and verify that the source has data. This fictional guide does not describe prices or an API.')]
])
print('Created two synthetic, two-page sample PDFs.')
