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

# YOUR personal policy: category -> (action, handling line the assistant learns).
POLICY: dict[str, tuple[str, str]] = {
    "newsletter": ("ARCHIVE", "ACTION: ARCHIVE"),
    "manager":    ("FLAG",    "ACTION: FLAG — surface to me, it's from my manager"),
    "meeting":    ("SCHEDULE","ACTION: SCHEDULE — offer Tue/Thu 2–4pm via calendly.com/me"),
    "invoice":    ("DELEGATE","ACTION: DELEGATE — forward to finance@acme.com"),
    "question":   ("REPLY",   "ACTION: REPLY — answer their question directly"),
}


def policy_action(email: dict) -> str:
    return POLICY[email["category"]][0]


def policy_target(email: dict) -> str:
    """The full assistant output your corrections teach for this email."""
    return POLICY[email["category"]][1]


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


# --- COACHING inbox: sender/subject pool A --------------------------------
COACH_EMAILS = [
    {"category": "newsletter", "sender": "digest@techweekly.com",
     "subject": "Your Tuesday tech digest", "body": "Top stories this week. Unsubscribe any time."},
    {"category": "manager", "sender": "priya@acme.com",
     "subject": "1:1 tomorrow", "body": "Can you prep a short update for our 1:1? — Priya (your manager)"},
    {"category": "meeting", "sender": "sam@partnerco.com",
     "subject": "Quick call?", "body": "Would love to find 30 minutes this week to sync. When are you free?"},
    {"category": "invoice", "sender": "billing@cloudhost.com",
     "subject": "Invoice #4471 due", "body": "Your monthly invoice of $240 is attached. Payment due in 14 days."},
    {"category": "question", "sender": "devon@acme.com",
     "subject": "Staging URL?", "body": "Hey, what's the URL for the staging environment again?"},
    {"category": "newsletter", "sender": "news@designtips.io",
     "subject": "5 layout tips", "body": "This week's design tips. Click unsubscribe to stop these."},
    {"category": "meeting", "sender": "hr@acme.com",
     "subject": "Schedule your review", "body": "Please book a slot for your quarterly review."},
    {"category": "invoice", "sender": "accounts@saasvendor.com",
     "subject": "Payment reminder", "body": "This is a reminder that invoice 88213 for billing is outstanding."},
    {"category": "manager", "sender": "priya@acme.com",
     "subject": "Roadmap", "body": "Need your input on the roadmap before Friday. — Priya"},
    {"category": "question", "sender": "casey@acme.com",
     "subject": "Deploy command?", "body": "How do I deploy to prod from this repo?"},
]

# --- HELD-OUT inbox: DISJOINT senders/subjects, same categories -----------
HELDOUT_EMAILS = [
    {"category": "invoice", "sender": "ar@datacenter.net",
     "subject": "Statement of charges", "body": "Your invoice for this month's billing is ready; payment due soon."},
    {"category": "manager", "sender": "priya@acme.com",
     "subject": "Headcount plan", "body": "Let's align on the headcount plan today. — Priya, your manager"},
    {"category": "meeting", "sender": "jordan@clientx.com",
     "subject": "Coffee next week?", "body": "Any chance we could grab time to talk through the proposal?"},
    {"category": "newsletter", "sender": "weekly@founderletters.com",
     "subject": "Founder notes", "body": "Your weekly founder reading. Unsubscribe at the bottom."},
    {"category": "question", "sender": "morgan@acme.com",
     "subject": "Where are the logs", "body": "Quick one — where do the service logs live?"},
    {"category": "meeting", "sender": "recruiter@talenthub.com",
     "subject": "Intro call", "body": "Would you have time to connect briefly this week?"},
    {"category": "invoice", "sender": "no-reply@toolsuite.com",
     "subject": "Receipt / amount due", "body": "Please find your billing statement; the amount due is $88."},
    {"category": "newsletter", "sender": "updates@marketpulse.co",
     "subject": "Market pulse", "body": "Daily market roundup. To stop, click unsubscribe."},
    {"category": "manager", "sender": "priya@acme.com",
     "subject": "Perf feedback", "body": "I have some feedback to share before your review. — Priya"},
    {"category": "question", "sender": "riley@acme.com",
     "subject": "API key rotation", "body": "How often should we rotate the API keys?"},
]
