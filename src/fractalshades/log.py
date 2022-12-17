# -*- coding: utf-8 -*-
import os
import datetime
import logging
import sys
import textwrap
import typing
import enum

import fractalshades as fs
# Default log levels
# CRITICAL 50
# ERROR 40
# WARNING 30
# INFO 20
# DEBUG 10
# NOTSET 0

# Log attributes
# https://docs.python.org/2/library/logging.html#logrecord-attributes

verbosity_list = (
    "warn @ console",
    "warn + info @ console",
    "debug @ console + log",
    "debug2 @ console + log",
)

verbosity_enum =  enum.Enum(
    "verbosity_enum",
    verbosity_list,
    module=__name__
)


def set_log_handlers(verbosity: typing.Literal[verbosity_enum]):
    """
    Sets the verbosity level for application logs.

    Parameters
    ----------
    verbosity: str
      Possible values for verbosity string parameter are :

        - "warn @ console" only warnings are printed to the console
        - "warn + info @ console" warnings and info are printed to the console
        - "debug @ console + log" warnings, info and debug level printed to 
          the console ; starts a new log file and outputs to it- same level
        - "debug2 @ console + log" same as above with lowest priority
          messages printed to log file.

    Notes
    -----
    The directory for the log files shall have been defined before
    through the `fractalshades.settings.log_directory` parameter.
    A typical use case is show below:

    ::
        
        fs.settings.log_directory = directory
        fs.set_log_handlers(verbosity="debug @ console + log")
    """
    if isinstance(verbosity, str):
        _verbosity = getattr(verbosity_enum, verbosity).value
    elif isinstance(verbosity, int):
        _verbosity = verbosity # Legacy: still accept int

    logger = logging.getLogger("fractalshades")

    # Remove previous handlers
    # https://stackoverflow.com/questions/12158048/changing-loggings-basicconfig-which-is-already-set
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Verbosity level mapping for console handler
    verbosity_mapping = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG,
    }
    logger.setLevel(verbosity_mapping[_verbosity])

    # create Console handler with a higher log level
    if _verbosity <= 0:
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


    # create File handler 
    if _verbosity >= 2:
        if fs.settings.log_directory is None:
            file_logger_warning = True
        else:
            file_logger_warning = False
            now = datetime.datetime.now()
            file_prefix = now.strftime("%Y-%m-%d_%Hh%M_%S")
            file_config = os.path.join(
                    fs.settings.log_directory,
                    f'{file_prefix}_factalshades.log'
            )

            # if directory for the log does not exists, creates it
            fs.utils.mkdir_p(os.path.dirname(file_config))

            fh = logging.FileHandler(file_config)
            fh.setLevel(logging.DEBUG)
            if _verbosity == 3:
                fh.setLevel(logging.NOTSET)
            fh_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s: %(funcName)s\n  "
                "%(message)s"
            )
            fh.setFormatter(fh_formatter)
            logger.addHandler(fh)


    logger.info(textwrap.dedent(f"""\
        =======================================
          Starting logger for fractalshades {fs.__version__}
          ======================================="""
    ))
    logger.info(f"Logger verbosity: {verbosity}")
    
    if  file_logger_warning:       
        logger.warning(
            "Unable to start file logger: "
            "fs.settings.log_directory not specified"
        )
    else:
        logger.info(
            f"Started file logger: {file_config}"
        )  

