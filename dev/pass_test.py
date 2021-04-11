#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:11:10 2021

@author: geoffroy
"""

def prompt_password(user):
    """
    Parameters
    ----------
    user : user name

    Returns
    -------
    text : user input password
    """
    from PyQt5.QtWidgets import  QInputDialog, QLineEdit, QApplication
    from PyQt5.QtCore import QCoreApplication

    # Let's avoid to crash Qt-based dev environment (Spyder...)
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication([])

    text, ok = QInputDialog.getText(
        None,
        "Credential",
        "user <{}>:".format(user),
        QLineEdit.Password)
    if ok and text:
        return text
    raise ValueError("Must specify a valid password")


if __name__ == "__main__":
    #test_view()
    pswd = prompt_password("username")
    print(pswd)
    #test_extended()