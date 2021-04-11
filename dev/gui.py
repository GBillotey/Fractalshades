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

import os
import fnmatch
import sys
import wx
import wx.lib.scrolledpanel as scrolled

from pubsub import pub

from perturbation import Perturbation_mandelbrot


MAIN_FOLDER = "/home/geoffroy/Pictures/math"


bg_not_validated_color = wx.Colour(255, 0, 0)
bg_validated_color = wx.Colour(188, 255, 188)


class DataFileValidator(wx.Validator):
    #https://stackoverflow.com/questions/2198903/wx-textctrl-and-wx-validator
    def __init__(self):
         wx.Validator.__init__(self)
    def Clone(self):
        return DataFileValidator()
    def TransferToWindow(self):
        return True
    def TransferFromWindow(self):
        return True
    def Validate(self, win):
        datatextCtrl = self.GetWindow()
        text = self.GetWindow().GetValue()
        file_path = os.path.join(wx.GetApp().main_folder, text)
        is_validated = os.path.isfile(file_path)
        if not(is_validated):
            datatextCtrl.SetBackgroundColour(bg_not_validated_color)
#            datatextCtrl.GetGrandParent().Refresh()
            return False
        else:

            
            data_dic = wx.GetApp().data_dic
            file_prefix
#            im.SetBitmap(wx.Bitmap(file_path))
#            im.Refresh()
            datatextCtrl.SetBackgroundColour(bg_validated_color)
            datatextCtrl.Refresh()
            return True


class ImageFileValidator(wx.Validator):
    #https://stackoverflow.com/questions/2198903/wx-textctrl-and-wx-validator
    def __init__(self):
         wx.Validator.__init__(self)
    def Clone(self):
        return ImageFileValidator()
    def TransferToWindow(self):
        return True
    def TransferFromWindow(self):
        return True
    def Validate(self, win):
        imtextCtrl = self.GetWindow()
        text = self.GetWindow().GetValue()
        file_path = os.path.join(wx.GetApp().main_folder, text)
        is_validated = os.path.isfile(file_path)
        if not(is_validated):
            imtextCtrl.SetBackgroundColour(bg_not_validated_color)
#            imtextCtrl.Refresh()
            return False
        else:
#            top_level = wx.GetTopLevelParent(imtextCtrl)
#            top_level.Freeze()TextCtrl
            pub.sendMessage("image.file", im_path=file_path)
#            im = wx.GetApp().image
#            im.SetBitmap(wx.Bitmap(file_path))
            imtextCtrl.SetBackgroundColour(bg_validated_color)
            imtextCtrl.Refresh()
#            top_level = wx.GetTopLevelParent(imtextCtrl)
#            top_level.Layout()
#            top_level.Update()
#            top_level.Thaw()
            return True

class MainFolderValidator(wx.Validator):
    """
    Validates the text control only if the text is a valid directory
    """
    def __init__(self):
         wx.Validator.__init__(self)
    def Clone(self):
        return MainFolderValidator()
    def TransferToWindow(self):
        return True
    def TransferFromWindow(self):
        return True
    def Validate(self, win):
        textCtrl = self.GetWindow()
        text = self.GetWindow().GetValue()
        is_validated = os.path.isdir(text)
        if not(is_validated):
            textCtrl.SetBackgroundColour(bg_not_validated_color)
            textCtrl.Refresh()
            return False
        else:
            wx.GetApp().main_folder = text
            textCtrl.SetBackgroundColour(bg_validated_color)
            textCtrl.Refresh()
            
            prefixes = get_file_prefixes(text)
            wx.GetApp().file_prefixes = prefixes
            
            # https://stackoverflow.com/questions/682923/dynamically-change-the-choices-in-a-wx-combobox
#            print("prefixes", prefixes)
#            staticbox = textCtrl.GetParent()
#            print("staticbox", staticbox)
#            data_text = staticbox.data_text
##            folderbox = staticbox.GetSizer()
##            print("folderbox", folderbox)
#            data_text.Clear()
#            for pre in prefixes:
#                data_text.Append(pre)
            
#            folderbox.data_text = wx.Choice(folderbox.GetStaticBox(),
#                choices=wx.GetApp().file_prefixes,
#                validator=DataFileValidator())
#            wx.GetTopLevelParent(textCtrl).Update()
            
            return True


#                def data_file(self, chunk_slice, file_prefix):
#        """
#        Returns the file path to store or retrieve data arrays associated to a 
#        data chunk
#        """
#        return os.path.join(self.directory, "data",
#                file_prefix + "_{0:d}-{1:d}_{2:d}-{3:d}.tmp".format(
#                *chunk_slice))
def get_file_prefixes(directory):
    """ Given a directory of fractal res files (.tmp) returns the list of the
    file prefixes
    """
    prefixes = []
    pattern = "*_*-*_*-*.tmp"
    data_dir = os.path.join(directory, "data")
    if not os.path.isdir(data_dir):
        return prefixes
    with os.scandir(data_dir) as it:
        for entry in it:
            if (fnmatch.fnmatch(entry.name, pattern) and
                os.path.relpath(os.path.dirname(entry.path), data_dir) == "."):
                frags = entry.name.split("_")
                candidate = ""
                for f in frags[:-2]:
                    candidate = candidate + f + "_"
                candidate = candidate[:-1]
                if candidate not in prefixes:
                    prefixes += [candidate]
    return prefixes


class FolderSizer(wx.StaticBoxSizer):
    def __init__(self, parent):
        super().__init__(orient=wx.VERTICAL, parent=parent,
             label="Main working folder and files")
        
        folder_group = wx.BoxSizer(wx.HORIZONTAL)
        self.folder_text = wx.TextCtrl(self.GetStaticBox(),
                                       value=wx.GetApp().main_folder,
                                       validator=MainFolderValidator())
        folder_button = wx.Button(self.GetStaticBox(), label="Main folder")
        folder_group.Add(self.folder_text, 1, wx.EXPAND|wx.BOTTOM|wx.LEFT, border=5)
        folder_group.Add(folder_button, 0, wx.EXPAND|wx.BOTTOM|wx.RIGHT, border=5)
        self.Add(folder_group, 0, wx.EXPAND|wx.BOTTOM|wx.RIGHT, border=0)

        image_group = wx.BoxSizer(wx.HORIZONTAL)
        self.image_text = wx.TextCtrl(self.GetStaticBox(), value="image",
                                       validator=ImageFileValidator())
        image_button = wx.Button(self.GetStaticBox(), label="image...")
        image_group.Add(self.image_text, 1, wx.EXPAND|wx.BOTTOM|wx.LEFT, border=5)
        image_group.Add(image_button, 0, wx.EXPAND|wx.BOTTOM|wx.RIGHT, border=5)
        self.Add(image_group, 0, wx.EXPAND|wx.BOTTOM|wx.RIGHT, border=0)
        
        data_group = wx.BoxSizer(wx.HORIZONTAL)
        self.data_text = wx.Choice(self.GetStaticBox(),
            choices=wx.GetApp().file_prefixes,
            validator=DataFileValidator())
        data_button = wx.Button(self.GetStaticBox(), label="data...")
        data_group.Add(self.data_text, 
                       1, wx.EXPAND|wx.BOTTOM|wx.LEFT, border=5)
        data_group.Add(data_button, 0, wx.EXPAND|wx.BOTTOM|wx.RIGHT, border=5)
        self.Add(data_group, 0, wx.EXPAND|wx.BOTTOM|wx.RIGHT, border=0)

        # event Binding
        folder_button.Bind(wx.EVT_BUTTON, self.folder_dialog)
        image_button.Bind(wx.EVT_BUTTON, self.image_dialog)
#        data_button.Bind(wx.EVT_BUTTON, self.data_dialog)
        self.folder_text.Bind(wx.EVT_TEXT, self.text_modified)
        self.image_text.Bind(wx.EVT_TEXT, self.text_modified)
        self.data_text.Bind(wx.EVT_CHOICE, self.data_choice)

        self.GetStaticBox().Validate()


    def folder_dialog(self, event):
        dlg = wx.DirDialog(self.GetStaticBox(), message="Choose main folder",
                            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
                            defaultPath=wx.GetApp().main_folder)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.folder_text.SetValue(path)
        dlg.Destroy()

    def image_dialog(self, event):
        with wx.FileDialog(self.GetStaticBox(), "Choose image file",
                           wildcard="PNG files (*.png)|*.png",
                           defaultDir=wx.GetApp().main_folder,
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:

            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            pathname = dlg.GetPath()
            self.image_text.SetValue(os.path.join("./", os.path.relpath(
                    pathname, start=wx.GetApp().main_folder)))
#
#    def data_dialog(self, event):
#        with wx.FileDialog(self.GetStaticBox(), "Choose data file",
#                           wildcard="TMP files (*.tmp)|*.tmp",
#                           defaultDir=wx.GetApp().main_folder,
#                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
#
#            if dlg.ShowModal() == wx.ID_CANCEL:
#                return
#            pathname = dlg.GetPath()
#            self.data_text.SetValue(os.path.join("./", os.path.relpath(
#                    pathname, start=wx.GetApp().main_folder)))

    def text_modified(self, event):
        self.GetStaticBox().Validate()
        prefixes = []#wx.GetApp().file_prefixes
        
        # https://stackoverflow.com/questions/682923/dynamically-change-the-choices-in-a-wx-combobox
        print("prefixes", prefixes)
#        staticbox = textCtrl.GetParent()
#        print("staticbox", staticbox)
        data_text = self.data_text
#            folderbox = staticbox.GetSizer()
#            print("folderbox", folderbox)
        data_text.Clear()
        for pre in prefixes:
            data_text.Append(pre)
            
        
    def data_choice(self, event):
        self.GetStaticBox().Validate()


class InfoSizer(wx.StaticBoxSizer):
    def __init__(self, parent):
        super().__init__(orient=wx.HORIZONTAL, parent=parent,
             label="Info display panel")


#class Dirchooser(wx.)
class ScrollbarPic(scrolled.ScrolledPanel):
    def __init__(self, parent):
        super().__init__(parent, style = wx.SUNKEN_BORDER)
        self.SetWindowStyleFlag(wx.SIMPLE_BORDER)
        self.bitmap=wx.StaticBitmap(parent=self)
        self.bitmap.SetBitmap(wx.Bitmap())

        self.imgSizer = wx.BoxSizer(wx.VERTICAL)   
        self.imgSizer.Add(self.bitmap, 1, wx.EXPAND)   
        self.SetSizer(self.imgSizer)

        #self.SetAutoLayout(1)
        self.SetupScrolling()

        self.bitmap.Bind(wx.EVT_MOUSE_EVENTS, self.OnMouse)
        
        self.IsRectReady = False
        self.newRectPara=[0,0,0,0]
        
        self.SetCursor(wx.StockCursor(wx.CURSOR_BULLSEYE))
        pub.subscribe(self.on_image_file, "image.file")

#        wx.GetApp().image = self.bitmap

    def OnMouse(self, event):
        dc = wx.ClientDC(self.bitmap)
        if event.Moving():
            pos = event.GetLogicalPosition(dc)
            print("pos", pos)
            
    def on_image_file(self, im_path):
        self.bitmap.SetBitmap(wx.Bitmap(im_path))
        self.SetupScrolling()
#        self.Parent().Layout()
#        self.Parent().Update()

#    def OnEnter(self, event):
#        print("enter")
#        myCursor= wx.StockCursor(wx.CURSOR_BULLSEYE)#CURSOR_CROSS)
#        self.SetCursor(myCursor)
        
#    def OnLeave(self, event):
#        print("leave")


            #self.scroll.SetScrollbars(1, 1, 1000, 1000)

#    def update_image(self, im_path):
#        png = wx.Image(im_path, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
#        wx.StaticBitmap(self, -1, png, (10, 5), (png.GetWidth(), png.GetHeight())




class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(800,600))
        #pic_path = "" #os.path.join("/home/geoffroy/Pictures/math/classic/test_order",
                         #       "divergent.png")
        self.scrollimage = ScrollbarPic(self) #, style=wx.TE_MULTILINE)
        self.folderbox = FolderSizer(self)
        self.zoombox = ZoomSizer(self)
        self.calcbox = CalcSizer(self)
        
        mainsizer = wx.BoxSizer(wx.HORIZONTAL)
        mainsizerleft = wx.BoxSizer(wx.VERTICAL)
        mainsizer.Add(self.scrollimage, 1, wx.EXPAND|wx.ALL, border=2)
        mainsizer.Add(mainsizerleft, 1, wx.EXPAND|wx.ALL, border=2)
        mainsizerleft.Add(self.folderbox, 0, wx.EXPAND|wx.ALL, border=2)  
        mainsizerleft.Add(self.infobox, 0, wx.ALL, border=2)  
        self.SetSizer(mainsizer)

        
        self.CreateStatusBar() # A StatusBar in the bottom of the window


        # Setting up the menu.
        filemenu= wx.Menu()

        # wx.ID_ABOUT and wx.ID_EXIT are standard ids provided by wxWidgets.
        menuAbout = filemenu.Append(wx.ID_ABOUT, "About"," Information about this program")
        menuExit = filemenu.Append(wx.ID_EXIT,"Exit"," Terminate the program")

        # Creating the menubr.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"File") # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.

        # Set events.
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)

        self.Show(True)        


    def OnAbout(self,e):
        # A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
        dlg = wx.MessageDialog( self, "A small text editor", "About Sample Editor", wx.OK)
        dlg.ShowModal() # Show it
        dlg.Destroy() # finally destroy it when finished.

    def OnExit(self,e):
        self.Close(True)  # Close the frame.



class Fractal_explore_app(wx.App):
    def __init__(self, main_folder):
        self.root_dic = {"folder": main_folder,
                         "fractal": Perturbation_mandelbrot}
#        self.data_dic = 
#        self.file_prefix = ""
#        self.file_prefixes = []
        super().__init__(redirect=False)
        #self.RestoreStdio()
        print("__init__ app")
        
    def OnInit(self) :
        frame = MainFrame(None, "Fractal viewer")
        frame.Show(True)
        self.SetTopWindow(frame)
        return True
    
    def default_data_dic(self):
        ref_fractal = Perturbation_mandelbrot
        return ref_fractal.gui_default_data_dic

def main():
    app = FractalApp("/home/geoffroy/Pictures/math") #wx.App(False)
    #frame = MainWindow(None, "Fractal viewer")
    app.MainLoop()
    
if "__main__" == __name__ :
    main()