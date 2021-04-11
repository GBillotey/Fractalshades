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

"""
A Pubsub_container
A Pubsub_item

event chain : when a contrl is modified it modifyes the underlying pubsubdict
   which in turn automatically trigger the message
"""

import os
import fnmatch
import sys
import wx
#import wx.lib.scrolledpanel as scrolled

from pubsub import pub

bg_not_validated_color = wx.Colour(255, 0, 0)
bg_validated_color = wx.Colour(188, 255, 188)

class Pubsub_dict():
    """
    Template:
    psd = Pubsub_dict({"a": {"position": ("main_panel", 1),
                             "type": "float",
                             "value": 1.,
                             "read_only": False,
                             "pubsub_func": None,
                             "proposed": None,
                             "validator": (func, kwargs_keys),
                             "suscribe": (topic, func},
    """
    def __init__(self, pub_dict, pub_name):
        self.pub_dict = pub_dict
        self.pub_name = pub_name

    def __setitem__(self, key, val):
        self.pub_dict[key] = val
        pub.sendMessage(self.pub_name + "." + key, newval=val)

    def __getitem__(self, key):
        return self.pub_dict[key]


class Pubsub_float(wx.TextCtrl):
    def __init__(self, parent, item_dic):
        super().__init(self, parent)
        self.value = item_dic.get("value")

    @property
    def value(self):
        return self.GetValue()

    @value.setter
    def value(self, val):
        self.GetValue(val)


class Pubsub_item_validator(wx.Validator):
    """
    Validates a Pubsub_item via its "validator" func
    """
    def __init__(self, func, kwargs_keys):
        self.func = func
        self.kwargs_keys = kwargs_keys
        wx.Validator.__init__(self)
    def Clone(self):
        return Pubsub_item_validator(self.func, self.kwargs_keys)
    def TransferToWindow(self):
        # https://www.pythonstudio.us/wxpython/how-do-i-use-a-validator-to-transfer-data.html
        return True
    def TransferFromWindow(self):
        return True
    def Validate(self, win):
        ctrl = self.GetWindow()
        psd = ctrl.psd
        values = list(psd[key]["value"] for key in self.kwargs_keys)
        kwargs = dict(zip(self.kwargs_keys, values))
        print("validate", kwargs)
        is_validated = self.func(**kwargs)
        ctrl.showstatus(validated=is_validated)
        return is_validated


class Pubsub_item(wx.Window):
    def __new__(cls, parent, item_dic, psub_dic):
        """
        Item dic : the item
        psd : the global dic
        """
        item_type = item_dic.get("type")
        value = item_dic.get("value")
        # Class factory...
        if item_type == "float":
            ret = Pubsub_float.__init__(parent, value)
        else:
            raise NotImplementedError(item_type)
        # Populating with useful data
        ret.psub_dic = psub_dic
        ret.item_dic = item_dic
        ret.suscribe()
        return ret

    def suscribe(self):
        subscriptions =  self.item_dic.get("subscriptions", None)
        if subscriptions is not None:
            for (topic, func) in subscriptions:
                print("adding subscription", topic, func)
                def updater(*args, **kwargs):
                    self.update(func(self.psd, *args, **kwargs))
                pub.subscribe(updater, topic)

    def add_validator(self):
        validator =  self.item_dic.get("validator", None)
        if validator is not None:
            func, psd_keys = validator
            print("adding validator", validator)
            self.SetValidator(Pubsub_item_validator(func, psd_keys))
            
    def showstatus(self,validated):
        choices = {True: bg_validated_color,
                   False:bg_not_validated_color}
        self.SetBackgroundColour(choices[validated])
        self.Refresh()

    @property
    def value(self):
        return NotImplementedError("Derived class must implements")
            
    @value.setter
    def value(self):
        return NotImplementedError("Derived class must implements")


class Pubsub_box(wx.StaticBoxSizer):
    def __init__(self, psd, orient=wx.HORIZONTAL):
        self.psd = psd
        self.group = wx.BoxSizer()


        
        
    def add_item(self, item_dic):
        item = Pubsub_item(item_dic, self.psd)
        self.group.Add(item_dic, 0, wx.EXPAND|wx.BOTTOM|wx.RIGHT, border=5)


#class PubsubMixin:
#    def __add__item
        

class Pubsub_frame(wx.Frame):
    """  wx.Frame class which implement a relationship beween a
    Pubsub_dict and the wx.Frame child GUI components
    """
    def __init__(self, psub_dic):
        super().__init__(self)
        self.psub_dic = psub_dic
#        self.register = dict()
        
    def setup_container(self, container, container_name):
        """ Container is a wx.window which will be attributed a name
        Container shall have a Add method
        """
#        self.register[container_name] = container
#        container.container_name = container_name
        # Populating container
        to_add = {}
        for item, i_dic in self.psub_dic.items():
            i_container, i_index = i_dic["position"]
            if i_container == container_name:
                to_add[i_index] = i_dic
        for added in sorted(to_add.keys()):
            container.add_item(i_dic)

#    def setup(self):
#        for key, item in self.psub_dic.items():
#            container, index = item.get("position")
            
#    def add_item(self, item_dic):
#        container, index = item_dic.get("position")
#        child = self.child_register[child_key]
#        child.add_item(item_dic, child_pos, self.psd)
        
    
    
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





def main():
    
    psub_dic = Pubsub_dict({"a": {"position": ("main_panel", 1),
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
    main()