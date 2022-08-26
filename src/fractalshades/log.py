# -*- coding: utf-8 -*-
import logging
import sys
import textwrap

# Default log levels
# CRITICAL 50
# ERROR 40
# WARNING 30
# INFO 20
# DEBUG 10
# NOTSET 0

# Log attributes
# https://docs.python.org/2/library/logging.html#logrecord-attributes

def set_log_handlers(verbosity, version_info=None):

    logger = logging.getLogger("fractalshades")
    verbosity_mapping = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logger.setLevel(verbosity_mapping[verbosity])

    # Remove previous handlers
    # https://stackoverflow.com/questions/12158048/changing-loggings-basicconfig-which-is-already-set
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if verbosity >= 2:
        # create file handler which logs debug messages
        fh = logging.FileHandler('session.log')
        fh.setLevel(logging.DEBUG)
        if verbosity == 3:
            fh.setLevel(logging.NOTSET)
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s: %(funcName)s\n  "
            "%(message)s"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    # create console handler with a higher log level
    if verbosity <= 0:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.WARNING)
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s\n  %(message)s'
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    if version_info is not None:
        logger.info(textwrap.dedent(f"""\
            ============================
              Loading fractalshades {version_info}
              ============================"""
        ))
        logger.info(f"Started logger with verbosity: {verbosity}")
    else:
        logger.info(f"Restarted logger with verbosity: {verbosity}")

    if verbosity >= 2:
        logger.info(f"Debugging log file: {fh}")
