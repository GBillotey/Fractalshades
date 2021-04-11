import numpy as np
import PIL
import os



def test_blender_normVEC():
#    The mode of an image defines the type and depth of a pixel in the image. The current release supports the following standard modes:
#L (8-bit pixels, black and white)
#P (8-bit pixels, mapped to any other mode using a color palette)
#RGB (3x8-bit pixels, true color)
#RGBA (4x8-bit pixels, true color with transparency mask)
#I (32-bit signed integer pixels)
#F (32-bit floating point pixels)
    mode = "RGB"
    plot_dir = r"/home/geoffroy/Images/Mes_photos/math/github_fractal_rep"
    file_name = "normVEC"
    
    kr = 20.
    taur = 2.
    kxy = 200.
    
    (x1, x2) = (-0.5, 1.)
    (y1, y2) = (-0.5, 1.)

    npts_x = 1200
    npts_y = 1200

    x = np.linspace(x1, x2, npts_x)
    y = np.linspace(y1, y2, npts_y)
    y, x  = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)

    z = 1. / kr * np.cos(kr * r) + 1. / kxy * np.cos(kxy * x) * np.cos(kxy * y)

    dzdr = - np.sin(kr * r)#* np.exp(-r * taur) + 
              #   -taur * np.cos(kr * r) * np.exp(-r * taur))
    
    dzdx = (x / r) * dzdr - np.sin(kxy * x) * np.cos(kxy * y)
    dzdy = (y / r) * dzdr - np.cos(kxy * x) * np.sin(kxy * y)
    dzdx = np.sin(kxy * x) * np.cos(kxy * y)
    dzdy = np.cos(kxy * x) * np.sin(kxy * y)
    rgb = np.zeros([npts_x, npts_y, 3], dtype=np.float32)
    rgb[:, :, 0] = dzdx
    rgb[:, :, 1] = dzdy
    rgb[:, :, 2] = -1.
    k = np.sqrt(rgb[:, :, 0]**2 + 
                rgb[:, :, 1]**2 + 
                rgb[:, :, 2]**2)
    print ("k", k, np.max(k))
    for ik in range(3):
        rgb[:, :, ik] = rgb[:, :, ik] / k
    rgb = 0.5 * (-rgb + 1.)
    rgb = np2PIL(rgb)
    rgb = np.uint8(rgb * 255)
    base_img = PIL.Image.fromarray(rgb, mode=mode)
    base_img.save(os.path.join(plot_dir, file_name + ".png"))


def test_blender_img():

    mode = "RGB"
    plot_dir = r"/home/geoffroy/Images/Mes_photos/math/github_fractal_rep"
    file_name = "color"

    (x1, x2) = (-0.5, 1.)
    (y1, y2) = (-0.5, 1.)

    npts_x = 1200
    npts_y = 1200

    x = np.linspace(x1, x2, npts_x)
    y = np.linspace(y1, y2, npts_y)
    y, x  = np.meshgrid(x, y)

    r = np.where((x % 0.1) < 0.01, 0., 1.)
    g = r*0. + 1.
    b = np.where((y % 0.1) < 0.01, 0., 1.)
    rgb = np.zeros([npts_x, npts_y, 3], dtype=np.float32)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    rgb = np2PIL(normalize(rgb))
    rgb = np.uint8(rgb * 255)
    base_img = PIL.Image.fromarray(rgb, mode=mode)
    base_img.save(os.path.join(plot_dir, file_name + ".png"))



def test_blender_dispVEC():
#    The mode of an image defines the type and depth of a pixel in the image. The current release supports the following standard modes:
#L (8-bit pixels, black and white)
#P (8-bit pixels, mapped to any other mode using a color palette)
#RGB (3x8-bit pixels, true color)
#RGBA (4x8-bit pixels, true color with transparency mask)
#I (32-bit signed integer pixels)
#F (32-bit floating point pixels)
    mode = "I"
    plot_dir = r"/home/geoffroy/Images/Mes_photos/math/github_fractal_rep"
    file_name = "dispVEC"
    
    kr = 20.
    taur = 2.
    kxy = 200.

    (x1, x2) = (-0.5, 1.)
    (y1, y2) = (-0.5, 1.)

    npts_x = 1200
    npts_y = 1200

    x = np.linspace(x1, x2, npts_x)
    y = np.linspace(y1, y2, npts_y)
    y, x  = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = 1. / kr * np.cos(kr * r)  + 1. / kxy * np.cos(kxy * x) * np.cos(kxy * y)

    greys = np2PIL(normalize(z))
    levels = np.int32(greys * (2**16 -1))
    base_img = PIL.Image.fromarray(levels, mode=mode)
    base_img.save(os.path.join(plot_dir, file_name + ".png"))


def np2PIL(arr):
    """
    Unfortunately this is a mess between numpy and pillow
    """
    sh = arr.shape
    if len(sh) == 2:
        nx, ny = arr.shape
        return np.swapaxes(arr, 0 , 1 )[::-1, :]
    nx, ny, _ = arr.shape
    return np.swapaxes(arr, 0 , 1 )[::-1, :, :]

def normalize(data, n11=False):
    """
    Renormalize data to [0;, 1.]
    """
    data_min, data_max = np.nanmin(data),  np.nanmax(data)
    data = (2 * data - (data_max + data_min)) / (
                    data_max - data_min)
    if not n11:
        data = 0.5 * (data + 1.)
    return data

if __name__ == "__main__":
    test_blender_dispVEC()
    test_blender_img()
    test_blender_normVEC()