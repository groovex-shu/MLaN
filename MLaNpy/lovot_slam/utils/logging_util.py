import logging


def setup_logging(syslog_identifier: str, debug: bool, journal: bool) -> None:
    modules_to_log = ('lovot_slam', 'lovot_map')
    loglevel = logging.DEBUG if debug else logging.INFO
    if journal:
        from systemd.journal import JournalHandler  # pylint: disable=import-error,import-outside-toplevel
        fmt = logging.Formatter("%(name)s %(levelname)s %(message)s")
        handler = JournalHandler(SYSLOG_IDENTIFIER=syslog_identifier)
        handler.setFormatter(fmt)
        for module in modules_to_log:
            logger = logging.getLogger(module)
            logger.addHandler(handler)
            logger.setLevel(loglevel)
    else:
        import coloredlogs
        for module in modules_to_log:
            logger = logging.getLogger(module)
            coloredlogs.install(level=loglevel, logger=logger)
