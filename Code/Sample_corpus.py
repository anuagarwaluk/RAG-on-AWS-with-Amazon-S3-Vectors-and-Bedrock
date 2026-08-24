"""Fictional HR policy corpus for the demo pipeline.

Small on purpose: the point of this repo is the architecture and the knobs,
not the data volume. Every document carries the metadata fields that make
the production patterns demonstrable: title/source/page for citations, and
tenant_id/access_group for retrieval-time security via metadata filters.
"""

DOCUMENTS = [
    {
        "doc_id": "hr-parental-leave",
        "title": "Parental Leave Policy",
        "source": "hr-handbook-2026.pdf",
        "page": 12,
        "tenant_id": "acme",
        "access_group": "all-employees",
        "text": (
            "Acme Corp provides 16 weeks of fully paid parental leave to primary "
            "caregivers and 6 weeks to secondary caregivers. Leave may be taken "
            "continuously or split into two blocks within the first 12 months "
            "after birth or adoption.\n\n"
            "Employees must notify their manager and HR at least 8 weeks before "
            "the expected start of leave. Benefits, including pension "
            "contributions and health cover, continue in full during parental "
            "leave. On return, employees are entitled to the same role or an "
            "equivalent position at the same grade."
        ),
    },
    {
        "doc_id": "hr-annual-leave",
        "title": "Annual Leave and Public Holidays",
        "source": "hr-handbook-2026.pdf",
        "page": 8,
        "tenant_id": "acme",
        "access_group": "all-employees",
        "text": (
            "Full time employees accrue 25 days of annual leave per year, plus "
            "public holidays observed in their country of employment. Up to 5 "
            "unused days may be carried into the first quarter of the following "
            "year with manager approval.\n\n"
            "Leave requests are submitted in the HR portal and approved by the "
            "line manager. Requests of 10 or more consecutive working days "
            "require 4 weeks notice. Annual leave cannot be exchanged for cash "
            "except on termination of employment, where accrued unused leave is "
            "paid out at the base salary rate."
        ),
    },
    {
        "doc_id": "hr-remote-work",
        "title": "Remote and Hybrid Working Policy",
        "source": "hr-handbook-2026.pdf",
        "page": 21,
        "tenant_id": "acme",
        "access_group": "all-employees",
        "text": (
            "Acme operates a hybrid model: employees work from an Acme office at "
            "least 2 days per week, with team anchor days set by each "
            "department. Fully remote arrangements are approved by exception "
            "for roles designated remote-eligible, and are reviewed annually.\n\n"
            "Employees working remotely must use company-managed devices, "
            "connect through the corporate VPN, and maintain a private "
            "workspace for calls involving confidential information. Working "
            "from another country for more than 20 days per year requires "
            "prior approval from HR and Tax due to residency implications."
        ),
    },
    {
        "doc_id": "hr-education",
        "title": "Education Reimbursement Programme",
        "source": "benefits-guide-2026.pdf",
        "page": 5,
        "tenant_id": "acme",
        "access_group": "all-employees",
        "text": (
            "Acme reimburses up to 5,000 per calendar year for approved "
            "external education, including certifications, university courses, "
            "and technical training relevant to the employee's current role or "
            "an agreed development path. Approval must be obtained before "
            "enrolment.\n\n"
            "Reimbursement is paid on evidence of successful completion. If an "
            "employee voluntarily leaves Acme within 12 months of receiving a "
            "reimbursement, 50 percent of the amount is repayable; within 6 "
            "months, 100 percent is repayable. Manager and HR approval records "
            "are retained for audit."
        ),
    },
    {
        "doc_id": "hr-expenses",
        "title": "Business Expenses Policy (Managers)",
        "source": "finance-policies-2026.pdf",
        "page": 3,
        "tenant_id": "acme",
        "access_group": "managers",
        "text": (
            "Managers may approve team expenses up to 500 per item without "
            "Finance review. Items above 500 require Finance pre-approval. "
            "Client entertainment is capped at 75 per head and must list "
            "attendees.\n\n"
            "All claims are submitted within 30 days with itemised receipts. "
            "Per diem rates apply for international travel and replace "
            "individual meal claims. Repeated late submissions are escalated "
            "to the budget holder. This document is restricted to the "
            "managers access group."
        ),
    },
]
