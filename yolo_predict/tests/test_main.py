from yolo_predict.__main__ import for_test
from yolo_predict.utils import for_test_2


def test_1():
    print("")
    for_test()
    assert True


def test_2():
    for_test_2()
    assert True
