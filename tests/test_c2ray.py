from pyc2ray.load_extensions import load_c2ray

c2ray = load_c2ray()


def test_load_c2ray():
    assert c2ray is not None
