import sys
major_version = sys.version_info[0]
if major_version == 3:
    from .ordered_enum import OrderedEnum
