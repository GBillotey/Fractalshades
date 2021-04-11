# -*- coding: utf-8 -*-
import PIL
import os

def main():
    directory = "/home/geoffroy/Pictures/math/classic/test_order"
    pic = "explore.png"
    Image = PIL.Image.open(os.path.join(directory, pic))
    print(Image.text)
    




if __name__ == "__main__":
    main()