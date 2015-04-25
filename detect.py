#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from scipy.ndimage import generic_filter
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import math 
import array

resource_dir = "resources/"
data = resource_dir+"dem.dat"

# todo figure out norms relative to flat plane
def load_file(file_name):
    with open(file_name, "rb") as f:
        array = np.fromfile(f, np.float32)
        array.byteswap(True)
        # array = np.flipud(np.reshape(array,(500,500)))
        return array.reshape(500,500)

surf1 = load_file("resources/Set1/surface.raw")
surf2 = load_file("resources/Set2/surface.raw")
surf3 = load_file("resources/Set3/surface.raw")
surf4 = load_file("resources/Set4/surface.raw")

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def pix_norm(mtx, pixel):
    x, y = pixel;
    x_scl = x * .2
    y_scl = y * .2

    # trying to get neighboring pxls for norms
    neighbors = [np.array ([nbrx*.2, nbry*.2,mtx[nbrx, nbry]])
                 for nbrx, nbry in [[x+1, y], [x-1, y]]]

    # getting mid pxl val
    origin = np.array ([x_scl,y_scl,mtx[x,y]])
    v1, v3, v2, v4 = [nbr - origin for nbr in neighbors]
    n1 = np.cross(v1,v2)
    print n1
    n2 = np.cross(v3,v4)

    z = [0,0,1]

    nga1 = py_ang(n1,z)
    a2 = py_ang(n2,z)
    return (a1+a2)/2


def circle_mask(N):
    max_d = ((N-1)//2)**2
    return np.array([[True if ((x-N//2)**2+(y-N//2)**2)
                      < max_d else False for x in range(N)]
             for y in range(N)])

def surf_grad(mtx):
    x_grad, y_grad = np.gradient(mtx)
    return x_grad + y_grad
    

zmat = np.array ([[0 for _ in range(500)] for _ in range(500)])

def bump_safe_mtx(mtx):
    resids = mtx - gaussian_filter(mtx,5)
    bumps = np.maximum(resids,zmat)
    t_mat = np.vectorize(lambda x: x>.35)(bumps)

    return t_mat

def obj_hazard(mtx):
    
    
    return 0
    
    
def window_slope(window):
    avg = np.average(window)
    window = window - avg
    window = window.reshape(17,17)
    mx = np.max(window)
    # if(mx)>.5:
    #     return 130
    rows, cols = window.shape
    x_m, y_m = rows//2, cols//2
    d_x, d_y = x_m*.2, y_m*.2

    z_m = window[x_m][y_m]
    # NE = (d_x, -d_y,window[0,cols-1])
    # NW = (-d_x, -d_y,window[0,0])
    # SW = (-d_x, d_y,window[rows-1,0])
    # SE = (d_x, d_y,window[rows-1,cols-1])
    # n3 = np.cross(NE,NW)
    # n4 = np.cross(SE,SW)
    # n_ = (n3+n4)/2

    N = (0,-d_y, window [x_m][0]-z_m)
    S = (0, d_y, window[x_m][rows-1]-z_m)
    E = (d_x, 0, window[cols-1][y_m]-z_m)
    W = (-d_x,0, window[0][y_m]-z_m)

    n1 = np.cross(N,E)
    n2 = np.cross(S,W)
    n = (n1+n2)/2
    # n = (n + n_)/2
    ang = py_ang(n,[0,0,1])*180/math.pi
    if ang > 90:
        ang = 180 - ang
        return ang < 10

def safe_slope_mtx(surf_mtx):
    # The max derivative that is within a 10 degree incline
    max_d = 0.2963269807

    surf_mtx = gaussian_filter(surf_mtx,2.1)
    rows, cols = surf_mtx.shape
    max_mtx = np.array([[max_d for _ in xrange(rows)] for _ in xrange(cols)])
    min_mtx = -max_mtx
    map_scl = [[.1 for _ in xrange(rows)] for _ in xrange(cols)]

    gx, gy = np.gradient(surf_mtx, map_scl, map_scl)

    x_safe = np.logical_and(np.less(gx, max_mtx),np.greater(gx, min_mtx))
    y_safe = np.logical_and(np.less(gy, max_mtx),np.greater(gy, min_mtx))
    gd = gx + gy
    d_safe = np.logical_and(np.less(gd, max_mtx*2),np.greater(gd, min_mtx*2))

    # y_safe = np.less(gy, max_mtx)
    safe_pts = np.logical_and(np.logical_and(x_safe, y_safe),d_safe)

    return safe_pts

def safe_to_pval(safe):
    if safe:
        return 255
    else: return 0

def numberize(safe_mtx):
    pfunc = np.vectorize(safe_to_pval)
    result =  pfunc(safe_mtx)
    return result

def output_pgm(safe_mtx):
    pfunc = np.vectorize(safe_to_pval)
    result =  pfunc(safe_mtx)
    return result.repeat(2,axis=0).repeat(2,axis=1).flatten()

def classify(mtx):
    slopes = safe_slope_mtx(mtx)
    puncs = safe_punc_mtx(mtx)
    safe = np.logical_and(slopes,puncs)
    return safe

def detect(mtx):
    slopes = safe_slope_mtx(mtx)
    puncs = safe_punc_mtx(mtx)
    safe = np.logical_and(slopes,puncs)
    return output_pgm(safe)

def safe_punc_mtx(mtx):
    bump_mtx = bump_safe_mtx(mtx)
    size = 9
    cm = circle_mask(size*2+1)
    rows, cols = mtx.shape

    clear = np.array([[False for _ in range (size*2+1)] for _ in range (size*2+1)])
    masks = np.array([[True for _ in range (rows)] for _ in range (cols)])
    for x, line in enumerate(bump_mtx):
        for y, bump in enumerate(line):
            if bump and x+size<rows and x-size >0 and y + size < rows and y-size > 0:
                window = bump_mtx[x-size:x+size+1,y-size:y+size+1]
                masked = mask_block(window,cm)
                # masks[x-size:x+size+1,y-size:y+size+1] = cm
                update= np.logical_and(masks[x+size:x+size+1,y-size:y+size+1],masked)
                masks[x-size:x+size+1,y-size:y+size+1] = update
                bump_mtx[x-size:x+size+1,y-size:y+size+1] = clear
                # np.logical_and(bump_mtx[x-15:x+16,
                #                         y-15:y+16],
                #                cm)

            
    return masks
def mask_at_index (mtx,index):
    x, y = index
    
def mask_block(block,mask):
    return np.logical_and(block,mask)

T = np.array([[_*2 for _ in range(50)] for _ in range(50)])

def flt(array):
    x = array.reshape(17,17)
    return x[1][1]
    # print array.shape

# it = np.nditer(T, flags=['multi_index'])
# while not it.finished:
#     # print "%d <%s>" % (it[0], it.multi_index),
#     x, y = it.multi_index
#     print "(%d %d)" % (x,y),
#     it.iternext()
def top_kek(mtx):
    output = generic_filter(mtx,window_slope,17,mode='constant')
    return output

derp = [True for _ in range(50)]
for i, x in enumerate(derp):
    if not i % 5:
        derp[i-3:i] = [False,False,False,False]
        np.fromstring

def view(mtx):
    plt.contourf(np.flipud(mtx))
    plt.show()

def viewT(mtx):
    plt.contourf(numberize(mtx))
    plt.show()

def viewPGM(mtx):
    omtx = output_pgm(mtx).reshape(1000,1000)
    plt.contourf(omtx)
    plt.show()
