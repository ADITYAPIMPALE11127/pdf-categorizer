# Create multiple PDF files using reportlab

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

def create_pdf(filename, title, content):
    doc = SimpleDocTemplate(f"/mnt/data/{filename}")
    elements = []
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))
    
    for para in content:
        elements.append(Paragraph(para, styles['Normal']))
        elements.append(Spacer(1, 10))
    
    doc.build(elements)

# 1. Service Agreement
create_pdf(
    "service_agreement.pdf",
    "Service Agreement - Wholesale Distributor",
    [
        "This Service Agreement is made between Distributor and Client.",
        "The Distributor agrees to supply goods in bulk quantities as per agreed schedule.",
        "Payment terms shall be Net 30 days from invoice date.",
        "Both parties agree to maintain confidentiality.",
        "This agreement is valid for 12 months unless terminated earlier."
    ]
)

# 2. Contract
create_pdf(
    "business_contract.pdf",
    "Wholesale Distribution Contract",
    [
        "This contract outlines the terms between Supplier and Distributor.",
        "Distributor shall purchase goods at agreed wholesale prices.",
        "Minimum order quantities must be maintained.",
        "Late payments may incur penalties.",
        "Disputes shall be resolved under applicable jurisdiction."
    ]
)

# 3. Court Order (Sample)
create_pdf(
    "court_order.pdf",
    "Sample Court Order - Business Dispute",
    [
        "This court order resolves a dispute between Supplier and Distributor.",
        "The Distributor is required to clear pending dues within 30 days.",
        "Failure to comply may result in legal enforcement.",
        "Both parties must adhere to the contractual obligations.",
        "Issued under authority of the court."
    ]
)

# 4. Business Report
create_pdf(
    "business_report.pdf",
    "Wholesale Business Report",
    [
        "This report summarizes monthly wholesale distribution performance.",
        "Total Revenue: ₹10,00,000",
        "Top Selling Categories: FMCG, Electronics, Groceries",
        "Operational Challenges: Logistics delays and inventory shortages",
        "Recommendations: Improve supply chain efficiency."
    ]
)

"/mnt/data/"