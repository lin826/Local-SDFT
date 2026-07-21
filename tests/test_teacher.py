from sdft.online.teacher import (
    DEFAULT_REINSTRUCT,
    build_student_messages,
    build_teacher_messages,
)


def test_teacher_appends_demonstration_and_reinstruct():
    msgs = [{"role": "user", "content": "How do I sort a list in Python?"}]
    teacher = build_teacher_messages(msgs, "Use sorted(xs) or xs.sort().")
    assert teacher[:-2] == msgs
    assert teacher[-2] == {"role": "assistant", "content": "Use sorted(xs) or xs.sort()."}
    assert teacher[-1] == {"role": "user", "content": DEFAULT_REINSTRUCT}


def test_teacher_does_not_mutate_input():
    msgs = [{"role": "user", "content": "hi"}]
    build_teacher_messages(msgs, "hello there")
    assert msgs == [{"role": "user", "content": "hi"}]


def test_student_is_plain_context():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    student = build_student_messages(msgs)
    assert student == msgs
    assert student is not msgs  # defensive copy


def test_custom_reinstruct():
    teacher = build_teacher_messages(
        [{"role": "user", "content": "q"}], "a", reinstruct="Try again."
    )
    assert teacher[-1]["content"] == "Try again."
