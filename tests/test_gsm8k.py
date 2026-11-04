from wsp import gsm8k


def test_sample_loads():
    rows = gsm8k.load()
    assert len(rows) == 50
    assert {"id", "question", "gold"} <= rows[0].keys()


def test_extract_boxed():
    assert gsm8k.extract_answer("so the total is \\boxed{1,250}.") == "1250"
    assert gsm8k.extract_answer("<think>maybe 3</think> The answer is \\boxed{$18}") == "18"


def test_extract_last_number_after_think():
    assert gsm8k.extract_answer("<think>7 or 9</think>She earns 18 dollars.") == "18"
    assert gsm8k.extract_answer("answer: 4.00") == "4"


def test_unfinished_think_has_no_answer():
    assert gsm8k.extract_answer("<think>18 18 18 18 18") is None


def test_is_correct():
    assert gsm8k.is_correct("\\boxed{70000}", "70000")
    assert not gsm8k.is_correct("\\boxed{7000}", "70000")
    assert not gsm8k.is_correct("no number here", "5")
