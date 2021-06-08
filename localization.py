import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import sys
from PIL import Image

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

calib_3d = sys.argv[1] + '/calib_3d.txt'
calib_2d = sys.argv[1] + '/calib_2d.txt'
calib_map = sys.argv[1] + '/calib_map.txt'
img_fname = sys.argv[1] + '/9.png'
label_fname = sys.argv[1] + '/prediction9.png'
map_fname = sys.argv[1] + '/map.png'
objp = np.loadtxt(calib_3d)
objp = np.hstack((objp, np.zeros((len(objp), 1)))).astype(np.float32)
imgp = np.loadtxt(calib_2d).astype(np.float32)
imgp = imgp.reshape(-1,1,2)

image_np = np.array(Image.open(img_fname))
gray = np.mean(image_np, axis=2)
mask = np.array(Image.open(label_fname))
print(image_np.shape, image_np.dtype, mask.shape, mask.dtype)
image_np[:,:,0] = gray
image_np[:,:,1] = gray
image_np[:,:,2] = gray
image_np[mask] = 0.5 * image_np[mask] + [0, 128, 0]
map_np = np.array(Image.open(map_fname))

_, mtx, _, rvecs, tvecs = cv.calibrateCamera([objp], [imgp], gray.shape[::-1], None, None, flags=cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_K1+cv.CALIB_FIX_K2+cv.CALIB_FIX_K3)
fx = mtx[0,0]
fy = mtx[1,1]
cx = mtx[0,2]
cy = mtx[1,2]
R, _ = cv.Rodrigues(rvecs[0])
T = tvecs[0]
print(fx,fy,cx,cy)
print(rvecs, R)
print(tvecs)
#h,  w = gray.shape
#newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#print(mtx, newcameramtx)
#undistorted = cv.undistort(gray, mtx, dist, None, newcameramtx)
#plt.figure()
#plt.imshow(gray, cmap='gray')
#plt.figure()
#plt.imshow(undistorted, cmap='gray')
#undistorted = cv.undistort(gray, mtx, dist, None, None)
#plt.figure()
#plt.imshow(undistorted, cmap='gray')
#plt.show()
imgp2, _ = cv.projectPoints(objp, rvecs[0], tvecs[0], mtx, np.zeros(5,dtype=np.float32))

min_cluster = 10
max_cluster = 1000
dt_2D = []
ret, dt_com = cv.connectedComponents(mask.astype(np.uint8))
for i in range(1, dt_com.max()+1):
    if np.sum(dt_com==i) > min_cluster and np.sum(dt_com==i) < max_cluster:
        my, mx = np.nonzero(dt_com==i)
        dt_2D.append([np.mean(mx), np.mean(my)])
dt_3D = np.zeros((len(dt_2D), 3), dtype=np.float32)
for i in range(len(dt_2D)):
    u,v = dt_2D[i]
    u = (u - cx) / fx
    v = (v - cy) / fy
    N = R.dot([0,0,1])
    z = N.dot(T) / (N[0]*u + N[1]*v + N[2])
    xyz = np.array([z*u, z*v, z])
    dt_3D[i,:] = R.T.dot(xyz - T).flatten()

plt.subplot(1,2,1)
plt.plot(imgp[:,:,0], imgp[:,:,1], 'ro')
plt.plot(imgp2[:,:,0], imgp2[:,:,1], 'bx')
plt.imshow(image_np)
for i in range(len(objp)):
    plt.annotate('(%d,%d)'%(objp[i,0],objp[i,1]), xy=(imgp[i,0,0], imgp[i,0,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
maxX = int(objp[:,0].max())
maxY = int(objp[:,1].max())
gridSize = 5
stepSize = 1
x = 0
while x <= maxX:
    pts = np.array([[x,y,0] for y in range(0, maxY+stepSize, stepSize)], dtype=np.float32)
    projected, _ = cv.projectPoints(pts, rvecs[0], tvecs[0], mtx, np.zeros(5,dtype=np.float32))
    plt.plot(projected[:,:,0], projected[:,:,1], 'k')
    x += gridSize
y = 0
while y <= maxY:
    pts = np.array([[x,y,0] for x in range(0, maxX+stepSize, stepSize)], dtype=np.float32)
    projected, _ = cv.projectPoints(pts, rvecs[0], tvecs[0], mtx, np.zeros(5,dtype=np.float32))
    plt.plot(projected[:,:,0], projected[:,:,1], 'k')
    y += gridSize

plt.subplot(1,2,2)
mapp = np.loadtxt(calib_map)
map_origin = mapp[0]
map_x_vector = mapp[1] - mapp[0]
map_y_vector = mapp[2] - mapp[0]
x = 0
while x <= maxX:
    p1 = map_origin + 1.0 * x / maxX * map_x_vector
    p2 = map_origin + map_y_vector + 1.0 * x / maxX * map_x_vector
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
    x += gridSize
y = 0
while y <= maxY:
    p1 = map_origin + 1.0 * y / maxY * map_y_vector
    p2 = map_origin + map_x_vector + 1.0 * y / maxY * map_y_vector
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
    y += gridSize
plt.imshow(map_np)
dt_map = np.zeros((len(dt_3D), 2))
for i in range(len(dt_3D)):
    dt_map[i] = map_origin + dt_3D[i,0] / maxX * map_x_vector + dt_3D[i,1] / maxY * map_y_vector
plt.plot(dt_map[:,0], dt_map[:,1], 'gx')
plt.annotate('(%d,%d)'%(0,0), xy=(mapp[0,0], mapp[0,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.annotate('(%d,%d)'%(maxX,0), xy=(mapp[1,0], mapp[1,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.annotate('(%d,%d)'%(0,maxY), xy=(mapp[2,0], mapp[2,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.annotate('(%d,%d)'%(maxX,maxY), xy=(mapp[3,0], mapp[3,1]), xytext=(20, 20), textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()
