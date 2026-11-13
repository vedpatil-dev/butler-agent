from stats import average, maximum


def test_average():
    assert average([2, 4, 6]) == 4


def test_maximum():
    assert maximum([1, 9, 3]) == 9
