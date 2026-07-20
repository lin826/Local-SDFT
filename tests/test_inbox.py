from sdft.online.inbox import (
    ACTIONS, COACH_EMAILS, HELDOUT_EMAILS, POLICY,
    format_email, parse_action, policy_action, policy_target,
)


def test_policy_covers_all_categories():
    cats = {e["category"] for e in COACH_EMAILS + HELDOUT_EMAILS}
    assert cats <= set(POLICY)
    assert all(POLICY[c][0] in ACTIONS for c in POLICY)


def test_heldout_disjoint_from_coach():
    ck = {(e["sender"], e["subject"]) for e in COACH_EMAILS}
    hk = {(e["sender"], e["subject"]) for e in HELDOUT_EMAILS}
    assert ck.isdisjoint(hk)


def test_parse_action():
    assert parse_action("ACTION: SCHEDULE — offer Tue/Thu") == "SCHEDULE"
    assert parse_action("action: archive") == "ARCHIVE"
    assert parse_action("I would FLAG this") == "FLAG"
    assert parse_action("no idea") is None


def test_policy_target_starts_with_action():
    for e in COACH_EMAILS:
        assert policy_target(e).upper().startswith("ACTION:")
        assert parse_action(policy_target(e)) == policy_action(e)


def test_format_email_contains_fields():
    e = COACH_EMAILS[0]
    txt = format_email(e)
    assert e["sender"] in txt and e["subject"] in txt and "handle" in txt
