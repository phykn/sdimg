import numpy as np

from sdimg.mask import fill_holes


def test_fill_holes_handles_empty_single_and_multi_holes() -> None:
    empty = np.zeros((5, 5), dtype=np.uint8)
    assert np.array_equal(fill_holes(empty), empty)

    # 단일 구멍이 있는 마스크
    single_hole = np.zeros((7, 7), dtype=np.uint8)
    single_hole[1:6, 1:6] = 1
    single_hole[3, 3] = 0
    assert np.count_nonzero(single_hole) == 24
    out1 = fill_holes(single_hole)
    assert np.count_nonzero(out1) == 25
    assert out1[3, 3] == 1

    # 다중 구멍이 있는 마스크
    multi_hole = np.zeros((7, 7), dtype=np.uint8)
    multi_hole[1:6, 1:6] = 1
    multi_hole[2, 2] = 0
    multi_hole[4, 4] = 0
    out2 = fill_holes(multi_hole)
    assert np.count_nonzero(out2) == 25
    assert out2[2, 2] == 1
    assert out2[4, 4] == 1
