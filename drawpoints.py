import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from PIL import Image
import random

from draw_data import a

x=[i[0] for i in a]
y=[i[1] for i in a]

def ext(idx):
    sz = 0.1
    return (x[idx]-sz, x[idx]+sz, y[idx]-sz, y[idx]+sz)

f=open("tsne.in0")
n=int(f.readline())
imgs = []
count = 0
for i in range(n):
    fname = f.readline()[:-1]
    if random.random() < 0.18 or 1:
        count+=1
        img = Image.open(fname)
        imgs.append((i,img))
random.shuffle(imgs)
fets = [i[0] for i in imgs]
imgs = [i[1] for i in imgs]

def min_resize(img, size):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    #w, h = map(float, img.shape[:2])
    w, h = img.size
    if min([w, h]) != size:
        if w <= h:
            img = img.resize((int(round((w/h)*size)), int(size)))
        else:
            img = img.resize((int(size), int(round((h/w)*size))))
    return img

def image_scatter(features, images, img_res, res=4000, cval=1.):
    """
    Embeds images via tsne into a scatter plot.
    Parameters
    ---------
    features: index list
        # Features to visualize
    images: list or numpy array
        Corresponding images to features. Expects float images from (0,1).
    img_res: float or int
        Resolution to embed images at
    res: float or int
        Size of embedding image in pixels
    cval: float or numpy array
        Background color value
    Returns
    ------
    canvas: numpy array
        Image of visualization
    """
    global x, y
    #features = np.copy(features).astype('float64')
    images = [image.convert("RGB") for image in images]
    images = [min_resize(image, img_res) for image in images]
    max_width = max([image.size[0] for image in images])
    max_height = max([image.size[1] for image in images])

    #f2d = bh_sne(features)

    #xx = f2d[:, 0]
    #yy = f2d[:, 1]
    xx = np.array([x[idx] for idx in features])
    yy = np.array([y[idx] for idx in features])
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    # Fix the ratios
    sx = (x_max-x_min)
    sy = (y_max-y_min)
    if sx > sy:
        res_x = int(sx/float(sy)*res)
        res_y = res
    else:
        res_x = res
        res_y = int(sy/float(sx)*res)

    canvas = np.ones((res_y+max_height, res_x+max_width, 3))*cval
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image in zip(xx, yy, images):
        w, h = image.size
        x_idx = np.argmin((x - x_coords)**2)
        y_idx = np.argmin((y - y_coords)**2)
        canvas[y_idx:y_idx+h, x_idx:x_idx+w] = np.array(image) / 256
    return canvas

canvas = image_scatter(fets, imgs, 250)

plt.imshow(canvas)
plt.show()
