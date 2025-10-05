import pytest

from lovot_slam.utils.immutable_mapping import ImmutableMapping


class ImmutableMappingWithMisssing(ImmutableMapping):
    def __missing__(self, key):
        return -100


def test_constructor():
    a = ImmutableMapping()
    assert a == {}

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert b == {'a': 1, 'b': 2}

    c = ImmutableMapping({})
    assert c == {}

    d = ImmutableMapping(b)
    assert d == {'a': 1, 'b': 2}
    assert d == b


def test_len():
    a = ImmutableMapping()
    assert len(a) == 0

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert len(b) == 2


def test_getitem():
    a = ImmutableMapping({'a': 1, 'b': 2})
    assert a['a'] == 1
    assert a['b'] == 2
    with pytest.raises(KeyError):
        a['c']

    b = ImmutableMappingWithMisssing({'a': 1, 'b': 2})
    assert b['a'] == 1
    assert b['b'] == 2
    assert b['c'] == -100


def test_get():
    a = ImmutableMapping({'a': 1, 'b': 2})
    assert a.get('a') == 1
    assert a.get('b') == 2
    assert a.get('c') is None
    assert a.get('c', -100) == -100

    b = ImmutableMappingWithMisssing({'a': 1, 'b': 2})
    assert b.get('a') == 1
    assert b.get('b') == 2
    assert b.get('c') == -100


def test_iter():
    a = ImmutableMapping()
    assert list(a) == []

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert list(b) == ['a', 'b']


def test_contains():
    a = ImmutableMapping()
    assert 'a' not in a

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert 'a' in b
    assert 'b' in b


def test_hash():
    a = ImmutableMapping()
    assert hash(a) == hash(tuple(sorted({}.items())))

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert hash(b) == hash(tuple(sorted({'a': 1, 'b': 2}.items())))

    c = ImmutableMapping({'a': 1, 'b': 2})
    d = ImmutableMapping({'a': 1, 'c': 2})
    assert hash(a) != hash(b)
    assert hash(a) != hash(c)
    assert hash(a) != hash(d)
    assert hash(b) == hash(c)
    assert hash(b) != hash(d)
    assert hash(c) != hash(d)


def test_repr():
    a = ImmutableMapping()
    assert repr(a) == "ImmutableMapping({})"

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert repr(b) == "ImmutableMapping({'a': 1, 'b': 2})"


def test_items():
    a = ImmutableMapping()
    assert list(a.items()) == []

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert list(b.items()) == [('a', 1), ('b', 2)]


def test_keys():
    a = ImmutableMapping()
    assert list(a.keys()) == []

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert list(b.keys()) == ['a', 'b']


def test_values():
    a = ImmutableMapping()
    assert list(a.values()) == []

    b = ImmutableMapping({'a': 1, 'b': 2})
    assert list(b.values()) == [1, 2]


def test_unsupported_methods():
    a = ImmutableMapping({'a': 1, 'b': 2})
    with pytest.raises(TypeError):
        a['a'] = 1

    with pytest.raises(TypeError):
        del a['a']

    with pytest.raises(AttributeError):
        a.clear()

    with pytest.raises(AttributeError):
        a.pop('a')

    with pytest.raises(AttributeError):
        a.popitem()

    with pytest.raises(AttributeError):
        a.setdefault('c', 1)

    with pytest.raises(AttributeError):
        a.update({'c': 1})
