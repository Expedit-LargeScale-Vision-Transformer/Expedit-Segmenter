import math


def get_aspect_ratio(x, y):
    gcd = math.gcd(x, y)
    return x // gcd, y // gcd


def reshape_as_aspect_ratio(x, ratio, channel_last=False):
    assert x.ndim == 3
    B, N, C = x.size()
    # print('size',ratio,x.size())
    s = round(math.sqrt(N / (ratio[0] * ratio[1])))
    perm = (0, 1, 2) if channel_last else (0, 2, 1)

    return x.permute(*perm).view(B, C, s * ratio[0], s * ratio[1])


def convert_len_to_hw(x, ratio):
    assert x.ndim == 3
    B, N, C = x.size()
    s = round(math.sqrt(N / (ratio[0] * ratio[1])))
    return s * ratio[0], s * ratio[1]
