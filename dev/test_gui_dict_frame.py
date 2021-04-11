# -*- coding: utf-8 -*-


### Installation note: how to install wx python
# 1) sudo apt-get install build-essential libgtk-3-dev
# 2) pip install wxPython==4.1.1 (takes a while to compile)
#
# https://pypubsub.readthedocs.io/en/v4.0.3/
#
# events:
# image.file
# image.mouse_move
# main_folder
# 

import wx
#import wx.lib.scrolledpanel as scrolled

from pubsub import pub

class Pubsub_dict():
    def __init__(self, pub_dict, pub_name):
        self.pub_dict = pub_dict
        self.pub_name = pub_name

    def __setitem__(self, key, val):
        self.pub_dict[key] = val
        pub.sendMessage(self.pub_name + "." + key, newval=val)

    def __getitem__(self, key):
        return self.pub_dict[key]


#class Gui_dict_panel(wx.Panel):
#    """
#    gui dict
#       key -> key_dict: {position:
#                         type:
#                         value:
#                         read_only: True/false
#                         pubsub: func
#                         proposed:
#                         validator:
#                         suscribe: [] list of keys
#                         }
#    """
#    def __init__(self, gui_dict)





def test_addfloat():
    
    psd = Pubsub_dict({"a": {"position": ("main_panel", 1),
                            "type": "float",
                            "value": 1.,
                            "read_only": False,
                            "pubsub_func": None,
                            "proposed": None,
                            "validator": None,
                            "suscribe": None},
                      "b": {"position": ("main_panel", 2),
                            "type": "float",
                            "value": 10.,
                            "read_only": False,
                            "pubsub_func": None,
                            "proposed": None,
                            "validator": None,
                            "suscribe": None},
                      "a + b": {"position": ("main_panel", 2),
                            "type": "float",
                            "value": 10.,
                            "read_only": False,
                            "pubsub_func": lambda: psd[a] + psd[b],
                            "proposed": None,
                            "validator": None,
                            "suscribe": ["a", "b"]}})
    
    
    
    app = wx.App(False)
    w = wx.Window()
    app.MainLoop()
    
if "__main__" == __name__ :

    class TestApp(wx.App):
        def __init__(self):
            super().__init__(redirect=False)
            
        def OnInit(self) :
            frame = MainWindow(None, "Fractal viewer")
            frame.Show(True)
            self.SetTopWindow(frame)
            return True

    class TestFrame(wx.Frame):
        def __init__(self, parent, title):
            wx.Frame.__init__(self, parent, title=title, size=(800,600))
            FolderSizer(wx.StaticBoxSizer)
        
    main()