# -*- coding: utf-8 -*-
import os
import datetime
import logging
import sys
import textwrap

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

def set_log_handlers(verbosity):
    """

    """
    logger = logging.getLogger("fractalshades")

    # Remove previous handlers
    # https://stackoverflow.com/questions/12158048/changing-loggings-basicconfig-which-is-already-set
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Sets verbosity level
    verbosity_mapping = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logger.setLevel(verbosity_mapping[verbosity])

    # create Console handler with a higher log level
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


    # create File handler 
    if verbosity >= 2:
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
            if verbosity == 3:
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

