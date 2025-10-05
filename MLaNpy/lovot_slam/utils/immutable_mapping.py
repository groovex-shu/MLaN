from typing import Iterable, Mapping, Optional, TypeVar

KT = TypeVar('KT')
VT = TypeVar('VT')


class ImmutableMapping(Mapping[KT, VT]):
    """Immutable mapping.
    Since immutable version of mapping (dictionary) is not implemented until Python 3.8,
    this class is used to make a mapping immutable.

    Reference: https://qiita.com/junkmd/items/7272a01bb0663e2138c9
    """
    def __init__(self, data: Optional[Mapping] = None) -> None:
        assert data is None or isinstance(data, Mapping)
        self._data = data if data is not None else {}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key: KT) -> VT:
        if key in self._data:
            return self._data[key]
        if hasattr(self.__class__, '__missing__'):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterable[KT]:
        return iter(self._data)

    def __contains__(self, key: KT) -> bool:
        return key in self._data

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items())))

    def __repr__(self) -> str:
        return f'ImmutableMapping({repr(self._data)})'
