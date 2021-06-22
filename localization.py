import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import sys
from PIL import Image
import argparse
import os

# convenience code to annotate calibration points by clicking
# python localization.py dataset/beach/map.png dataset/beach/calib_map.txt --click
if '--click' in sys.argv:
    input_fname = sys.argv[1]
    output_fname = sys.argv[2]
    img = cv.imread(input_fname)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fig = plt.figure()

    clicked_points = []
    def update():
        plt.clf()
        plt.imshow(img, cmap='gray')
        plt.plot([p[0] for p in clicked_points], [p[1] for p in clicked_points], 'rx')
        fig.canvas.draw()

    def onclick(event):
        x,y = int(np.round(event.xdata)), int(np.round(event.ydata))
        print('clicked',x,y)
        clicked_points.append([x, y])
        update()

    def onkey(event):
        if event.key==' ':
            np.savetxt(output_fname, clicked_points)
            sys.exit(1)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    update()
    plt.show()
    sys.exit(1)

# code to project 2D detections on to an overhead map

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="beach", required=True, help="Dataset directory name")
parser.add_argument( '--map', type=str, required=True, help='satellite image')
parser.add_argument("--gridSize", type=int, default=10, help="size of each square in the calibration grid")
parser.add_argument("--stepSize", type=int, default=1, help="density of drawing calibration grid")
flags = parser.parse_args()

# visualize calibration points in overhead map
plt.figure()
map_ax = plt.gca()
# oriented bounding box of calibration area in overhead map
calib_map = flags.map.replace('.png', '.txt')
calib_map = np.loadtxt(calib_map)
map_origin = calib_map[0]
map_x_vector = calib_map[1] - calib_map[0]
map_y_vector = calib_map[2] - calib_map[0]
maxX = 330
maxY = 130
x = 0
while x <= maxX:
    p1 = map_origin + 1.0 * x / maxX * map_x_vector
    p2 = map_origin + map_y_vector + 1.0 * x / maxX * map_x_vector
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k', linewidth=1, alpha=0.5)
    x += flags.gridSize
y = 0
while y <= maxY:
    p1 = map_origin + 1.0 * y / maxY * map_y_vector
    p2 = map_origin + map_x_vector + 1.0 * y / maxY * map_y_vector
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k', linewidth=1, alpha=0.5)
    y += flags.gridSize
# image of overhead map / satellite image
map_fname = flags.map
map_np = np.array(Image.open(map_fname))
plt.imshow(map_np)
plt.annotate('(%d,%d)'%(0,0), xy=(calib_map[0,0], calib_map[0,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.annotate('(%d,%d)'%(maxX,0), xy=(calib_map[1,0], calib_map[1,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.annotate('(%d,%d)'%(0,maxY), xy=(calib_map[2,0], calib_map[2,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.annotate('(%d,%d)'%(maxX,maxY), xy=(calib_map[3,0], calib_map[3,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.annotate('CCTV #2, #3, #4\nLAT=38.1899232\nLON=128.6040349', xy=(258, 424), xytext=(-30, -30), textcoords='offset points', ha='right', va='top',
    bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.annotate('CCTV #1, #5, #6\nLAT=38.1893355\nLON=128.6047293', xy=(367, 583), xytext=(-30, -30), textcoords='offset points', ha='right', va='top',
    bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

for dataset in flags.datasets.split(','):

    print('Loading from', dataset)
    # calibration points in world coordinates (assume Z coordinate of 0)
    calib_3d = dataset + '/calib_3d.txt'
    # corresponding calibration point coordinates in the input image
    calib_2d = dataset + '/calib_2d.txt'
    calib_3d = np.loadtxt(calib_3d)
    calib_3d = np.hstack((calib_3d, np.zeros((len(calib_3d), 1)))).astype(np.float32)
    calib_2d = np.loadtxt(calib_2d).astype(np.float32)
    calib_2d = calib_2d.reshape(-1,1,2)

    img_idx = 1
    while True:
        # input image
        img_fname = dataset + '/%d.png' % img_idx
        # input semantic segmentation mask
        label_fname = dataset + '/prediction%d.png' % img_idx
        if os.path.exists(label_fname):
            break
        img_idx += 1
    image_np = np.array(Image.open(img_fname))
    gray = np.mean(image_np, axis=2)
    mask = np.array(Image.open(label_fname))
    print('Input', image_np.shape, image_np.dtype, mask.shape, mask.dtype)
    image_np[:,:,0] = gray
    image_np[:,:,1] = gray
    image_np[:,:,2] = gray
    image_np[mask] = 0.5 * image_np[mask] + [0, 128, 0]

    # solve for camera parameters
    _, mtx, _, rvecs, tvecs = cv.calibrateCamera([calib_3d], [calib_2d], gray.shape[::-1], None, None, flags=cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_K1+cv.CALIB_FIX_K2+cv.CALIB_FIX_K3)
    fx = mtx[0,0]
    fy = mtx[1,1]
    cx = mtx[0,2]
    cy = mtx[1,2]
    R, _ = cv.Rodrigues(rvecs[0])
    T = tvecs[0]
    print('Camera parameters:')
    print(fx,fy,cx,cy)
    print(rvecs, R)
    print(tvecs)
    calib_2d_reprojected, _ = cv.projectPoints(calib_3d, rvecs[0], tvecs[0], mtx, np.zeros(5,dtype=np.float32))

    # get detection centroids from semantic segmentation mask
    min_cluster = 10
    max_cluster = 1000
    dt_2D = []
    ret, dt_com = cv.connectedComponents(mask.astype(np.uint8))
    for i in range(1, dt_com.max()+1):
        if np.sum(dt_com==i) > min_cluster and np.sum(dt_com==i) < max_cluster:
            my, mx = np.nonzero(dt_com==i)
            dt_2D.append([np.mean(mx), np.mean(my)])

    # project detection centroids to world coordinates at Z=0
    dt_3D = np.zeros((len(dt_2D), 3), dtype=np.float32)
    for i in range(len(dt_2D)):
        u,v = dt_2D[i]
        u = (u - cx) / fx
        v = (v - cy) / fy
        N = R.dot([0,0,1])
        z = N.dot(T) / (N[0]*u + N[1]*v + N[2])
        xyz = np.array([z*u, z*v, z])
        dt_3D[i,:] = R.T.dot(xyz - T).flatten()

    # visualize calibration points and detection centroids in input image
    plt.figure()
    plt.plot(calib_2d[:,:,0], calib_2d[:,:,1], 'ro')
    plt.plot(calib_2d_reprojected[:,:,0], calib_2d_reprojected[:,:,1], 'bx')
    plt.imshow(image_np)
    for i in range(len(calib_3d)):
        plt.annotate('(%d,%d)'%(calib_3d[i,0],calib_3d[i,1]), xy=(calib_2d[i,0,0], calib_2d[i,0,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    x1 = int(np.floor(1.0 * calib_3d[:,0].min() / flags.gridSize)) * flags.gridSize
    x2 = int(np.ceil(1.0 * calib_3d[:,0].max() / flags.gridSize)) * flags.gridSize
    y1 = int(np.floor(1.0 * calib_3d[:,1].min() / flags.gridSize)) * flags.gridSize
    y2 = int(np.ceil(1.0 * calib_3d[:,1].max() / flags.gridSize)) * flags.gridSize
    # draw grid lines
    x = x1
    while x <= x2:
        pts = np.array([[x,y,0] for y in range(y1, y2+flags.stepSize, flags.stepSize)], dtype=np.float32)
        projected, _ = cv.projectPoints(pts, rvecs[0], tvecs[0], mtx, np.zeros(5,dtype=np.float32))
        plt.plot(projected[:,:,0], projected[:,:,1], 'k')
        x += flags.gridSize
    y = y1
    while y <= y2:
        pts = np.array([[x,y,0] for x in range(x1, x2+flags.stepSize, flags.stepSize)], dtype=np.float32)
        projected, _ = cv.projectPoints(pts, rvecs[0], tvecs[0], mtx, np.zeros(5,dtype=np.float32))
        plt.plot(projected[:,:,0], projected[:,:,1], 'k')
        y += flags.gridSize

    # add detection centroids to overhead map
    dt_map = np.zeros((len(dt_3D), 2))
    for i in range(len(dt_3D)):
        dt_map[i] = map_origin + dt_3D[i,0] / maxX * map_x_vector + dt_3D[i,1] / maxY * map_y_vector
    map_ax.plot(dt_map[:,0], dt_map[:,1], 'x', linewidth=3, label='CCTV'+dataset.split('_')[-1])

map_ax.legend()
plt.show()
