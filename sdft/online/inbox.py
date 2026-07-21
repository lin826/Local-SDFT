"""A realistic "learn while doing" scenario: a personal inbox assistant.

As you clear your morning inbox, the assistant proposes how to handle each email
(archive / flag / schedule / delegate / reply) in *your* style. You fix the ones
it gets wrong. Within a few corrections it has learned your personal triage
policy + handling and applies it to the rest of the inbox — and to tomorrow's
new mail.

Why this is a genuine online-learning use case (not RAG):
- It's a learned *decision policy* over your mail, not a fact lookup.
- Your policy is personal (you FLAG your manager; someone else might not) and
  private (never leaves the device).
- It must generalize to emails you've never seen — new senders, new subjects.

For an objective, reproducible demo, each email carries a hidden `category`, and
your policy maps category -> handling. The model only sees sender/subject/body,
so it must infer the category and recall your policy. Held-out emails use a
sender/subject pool DISJOINT from coaching, so success can't be memorization.
"""

from __future__ import annotations

import re

ACTIONS = ["ARCHIVE", "FLAG", "SCHEDULE", "DELEGATE", "REPLY"]

# YOUR personal policy: category -> action. (A different user's policy could map
# these differently; the model learns *yours* from your corrections.)
POLICY: dict[str, str] = {
    "newsletter": "ARCHIVE",
    "manager":    "FLAG",
    "meeting":    "SCHEDULE",
    "invoice":    "DELEGATE",
    "question":   "REPLY",
}


def policy_action(email: dict) -> str:
    return POLICY[email["category"]]


def policy_target(email: dict) -> str:
    """The assistant output your corrections teach for this email (short + crisp)."""
    return f"ACTION: {POLICY[email['category']]}"


def format_email(email: dict) -> str:
    return (f"From: {email['sender']}\n"
            f"Subject: {email['subject']}\n\n"
            f"{email['body']}\n\n"
            f"How should I handle this email?")


def parse_action(reply: str) -> str | None:
    """Extract the ACTION tag the model produced (case-insensitive)."""
    m = re.search(r"ACTION:\s*([A-Za-z]+)", reply or "", re.IGNORECASE)
    if m:
        tag = m.group(1).upper()
        return tag if tag in ACTIONS else None
    # fall back: a bare action word appearing in the reply
    for a in ACTIONS:
        if re.search(rf"\b{a}\b", (reply or "").upper()):
            return a
    return None


# --- COACHING inbox: sender/subject pool A (6 per category) ---------------
COACH_EMAILS = [
    # newsletter -> ARCHIVE
    {"category": "newsletter", "sender": "digest@techweekly.com", "subject": "Your Tuesday tech digest",
     "body": "Top stories this week. Unsubscribe any time."},
    {"category": "newsletter", "sender": "news@designtips.io", "subject": "5 layout tips",
     "body": "This week's design tips. Click unsubscribe to stop these emails."},
    {"category": "newsletter", "sender": "hello@productmonthly.com", "subject": "Product monthly",
     "body": "Your monthly roundup of product news. Manage preferences or unsubscribe below."},
    {"category": "newsletter", "sender": "team@devbytes.dev", "subject": "DevBytes weekly",
     "body": "This week in engineering. You are receiving this newsletter because you subscribed. Unsubscribe."},
    {"category": "newsletter", "sender": "list@growthhackers.co", "subject": "Growth digest",
     "body": "Weekly growth reads. To opt out, unsubscribe here."},
    {"category": "newsletter", "sender": "noreply@bookclub.org", "subject": "New picks",
     "body": "This month's reading list. Unsubscribe if you'd rather not receive these."},
    # manager -> FLAG
    {"category": "manager", "sender": "priya@acme.com", "subject": "1:1 tomorrow",
     "body": "Can you prep a short update for our 1:1? — Priya, your manager"},
    {"category": "manager", "sender": "priya@acme.com", "subject": "Roadmap input",
     "body": "Need your input on the roadmap before Friday. — Priya (manager)"},
    {"category": "manager", "sender": "priya@acme.com", "subject": "Quick favor",
     "body": "As your manager I'd like you to take the lead on this. — Priya"},
    {"category": "manager", "sender": "priya@acme.com", "subject": "Team goals",
     "body": "Let's review your goals for the quarter. — Priya, your manager"},
    {"category": "manager", "sender": "priya@acme.com", "subject": "Priority shift",
     "body": "I'm reprioritizing your work this sprint. — Priya (your manager)"},
    {"category": "manager", "sender": "priya@acme.com", "subject": "Kudos",
     "body": "Great job last week. Your manager, Priya."},
    # meeting -> SCHEDULE
    {"category": "meeting", "sender": "sam@partnerco.com", "subject": "Quick call?",
     "body": "Would love to find 30 minutes this week to sync. When are you free?"},
    {"category": "meeting", "sender": "hr@acme.com", "subject": "Schedule your review",
     "body": "Please book a slot for your quarterly review meeting."},
    {"category": "meeting", "sender": "lee@vendorplus.com", "subject": "Demo time",
     "body": "Can we set up a call to walk through the demo? Let me know your availability."},
    {"category": "meeting", "sender": "alex@partnerco.com", "subject": "Sync",
     "body": "Let's find time to meet and align on next steps."},
    {"category": "meeting", "sender": "events@confhub.com", "subject": "Speaker call",
     "body": "We'd like to schedule a brief call to discuss your talk. What times work?"},
    {"category": "meeting", "sender": "dana@clientco.com", "subject": "Catch up",
     "body": "Are you free to meet sometime this week to catch up on the project?"},
    # invoice -> DELEGATE
    {"category": "invoice", "sender": "billing@cloudhost.com", "subject": "Invoice #4471 due",
     "body": "Your monthly invoice of $240 is attached. Payment due in 14 days."},
    {"category": "invoice", "sender": "accounts@saasvendor.com", "subject": "Payment reminder",
     "body": "Invoice 88213 for billing is outstanding. Please arrange payment."},
    {"category": "invoice", "sender": "ar@hostingpro.com", "subject": "Amount due",
     "body": "Attached is your billing statement; the amount due is $512."},
    {"category": "invoice", "sender": "finance@officeco.com", "subject": "Receipt",
     "body": "Your invoice and receipt for this month's charges are enclosed."},
    {"category": "invoice", "sender": "billing@dataworks.io", "subject": "Overdue",
     "body": "Your payment for invoice 55-A is overdue. Please remit the balance."},
    {"category": "invoice", "sender": "pay@subscriptions.net", "subject": "Charge",
     "body": "We've issued an invoice for your subscription; payment is due."},
    # question -> REPLY
    {"category": "question", "sender": "devon@acme.com", "subject": "Staging URL?",
     "body": "Hey, what's the URL for the staging environment again?"},
    {"category": "question", "sender": "casey@acme.com", "subject": "Deploy command?",
     "body": "How do I deploy to prod from this repo?"},
    {"category": "question", "sender": "jamie@acme.com", "subject": "Config file",
     "body": "Which config file controls the timeouts?"},
    {"category": "question", "sender": "pat@acme.com", "subject": "Quick q",
     "body": "Do you know who owns the billing service?"},
    {"category": "question", "sender": "sky@acme.com", "subject": "Test data",
     "body": "Where can I find the test fixtures for the parser?"},
    {"category": "question", "sender": "quinn@acme.com", "subject": "Version",
     "body": "What version of Python does the project target?"},
]

# --- HELD-OUT inbox: DISJOINT senders/subjects, same categories (3/cat) ----
HELDOUT_EMAILS = [
    {"category": "invoice", "sender": "ar@datacenter.net", "subject": "Statement of charges",
     "body": "Your invoice for this month's billing is ready; payment due soon."},
    {"category": "manager", "sender": "priya@acme.com", "subject": "Headcount plan",
     "body": "Let's align on the headcount plan today. — Priya, your manager"},
    {"category": "meeting", "sender": "jordan@clientx.com", "subject": "Coffee next week?",
     "body": "Any chance we could grab time to meet and talk through the proposal?"},
    {"category": "newsletter", "sender": "weekly@founderletters.com", "subject": "Founder notes",
     "body": "Your weekly founder reading. Unsubscribe at the bottom."},
    {"category": "question", "sender": "morgan@acme.com", "subject": "Where are the logs",
     "body": "Quick one — where do the service logs live?"},
    {"category": "meeting", "sender": "recruiter@talenthub.com", "subject": "Intro call",
     "body": "Would you have time to schedule a brief call to connect this week?"},
    {"category": "invoice", "sender": "no-reply@toolsuite.com", "subject": "Amount due",
     "body": "Please find your billing statement; the amount due is $88."},
    {"category": "newsletter", "sender": "updates@marketpulse.co", "subject": "Market pulse",
     "body": "Daily market roundup. To stop these, click unsubscribe."},
    {"category": "manager", "sender": "priya@acme.com", "subject": "Perf feedback",
     "body": "As your manager I have feedback to share before your review. — Priya"},
    {"category": "question", "sender": "riley@acme.com", "subject": "API key rotation",
     "body": "How often should we rotate the API keys?"},
    {"category": "invoice", "sender": "collections@webscale.io", "subject": "Past due",
     "body": "Your invoice remains unpaid; the outstanding payment is now past due."},
    {"category": "newsletter", "sender": "hi@startupdaily.com", "subject": "Daily brief",
     "body": "Your startup daily brief. Unsubscribe any time from the footer."},
    {"category": "question", "sender": "sage@acme.com", "subject": "Env var",
     "body": "Which environment variable sets the log level?"},
    {"category": "meeting", "sender": "chris@bizdev.co", "subject": "Partnership chat",
     "body": "Could we meet to discuss a possible partnership? When works for you?"},
    {"category": "manager", "sender": "priya@acme.com", "subject": "Sprint check-in",
     "body": "Let's do a quick check-in on the sprint. — Priya, your manager"},
]
