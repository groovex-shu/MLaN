class SlamError(Exception):
    pass


class SlamProcessError(SlamError):
    pass


class SlamMapError(SlamError):
    pass


class SlamTransferError(SlamError):
    pass


class SlamProcedureCallError(SlamError):
    pass


class SlamSensorError(SlamError):
    pass


class SlamBuildMapError(SlamError):
    pass
