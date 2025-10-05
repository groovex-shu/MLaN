from enum import Enum
from functools import total_ordering


@total_ordering
class OrderedEnum(Enum):
    """Enum supporting comparison based on underlying type."""

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            # https://github.com/PyCQA/pylint/issues/2306
            return self.value < other.value  # pylint: disable=comparison-with-callable
        return NotImplemented
