import matplotlib.pyplot as plt
import numpy
from PIL import Image
from sklearn.cluster import KMeans
import sys
import os

dataset = 'beach'
margin = 10
use_rgb = True
img_id=1
prelabel = None
mode = 'light'
print('Light mode')
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--dataset':
		dataset = sys.argv[i+1]
	elif sys.argv[i]=='--margin':
		margin = int(sys.argv[i+1])
	elif sys.argv[i]=='--id':
		img_id = int(sys.argv[i+1])
	elif sys.argv[i]=='--prelabel':
		prelabel = numpy.array(Image.open(sys.argv[i+1])) > 0
#fig = plt.figure()
fig = plt.figure(figsize=(20,30))

def next_img():
	global image_np, image_display, labels, img_id, segment_id
	image_filename = 'dataset/%s/%d.png'%(dataset,img_id)
	if os.path.exists(image_filename):
		image_np = numpy.array(Image.open(image_filename))
	else:
		sys.exit(0)
	image_display = image_np.copy()
	if not use_rgb:
		image_np = numpy.mean(image_np, axis=2)
#	print(image_np.shape, image_display.shape)
	if prelabel is None:
		labels = numpy.zeros((image_np.shape[0], image_np.shape[1]), dtype=int)
		segment_id=0
	else:
		labels = prelabel.astype(int)
		image_display[labels>0] = [255,0,0]
		segment_id=1
	plt.imshow(image_display)
	print('Image #%d Segment #%d'%(img_id, segment_id))
	plt.title('Image #%d Segment #%d'%(img_id, segment_id))
	fig.canvas.draw()

def onkey(event):
	if event.key==' ':
		global img_id, labels
		labels = (labels>0) * 255
		print(labels.dtype, set(labels.flatten()))
		Image.fromarray(labels.astype(numpy.uint8), mode='L').save('dataset/%s/label%d.png'%(dataset,img_id))
		img_id += 1
		next_img()
	elif event.key=='r':
		next_img()
	elif event.key=='u':
		undo()
	elif event.key=='l':
		global mode
		if mode=='light':
			mode = 'dark'
			print('Dark mode')
		elif mode=='dark':
			mode = 'light'
			print('Light mode')
		plt.clf()
		plt.imshow(image_display)
		fig.canvas.draw()

def undo():
	global segment_id
	if segment_id==0:
		return
	previous_mask = labels==segment_id
	labels[previous_mask] = 0
	image_display[previous_mask] = image_np[previous_mask]
	segment_id -= 1
	plt.clf()
	plt.imshow(image_display)
	print('Undo segment #%d' % (segment_id))
	fig.canvas.draw()

def onclick(event):
	global segment_id
	x,y = int(numpy.round(event.xdata)), int(numpy.round(event.ydata))
	xl = max(0,x-margin)
	xr = min(image_np.shape[1],x+margin)
	yl = max(0,y-margin)
	yr = min(image_np.shape[0],y+margin)
	cropped = image_np[yl:yr, xl:xr]
	if use_rgb:
		kmeans = KMeans(n_clusters=2).fit(cropped.reshape(-1,3))
		print('%.2f (%d) %.2f (%d)'%(kmeans.cluster_centers_[0][0], numpy.sum(kmeans.labels_==0), kmeans.cluster_centers_[1][0], numpy.sum(kmeans.labels_==1)))
		if mode=='light':
			target_label = numpy.argmax(kmeans.cluster_centers_.mean(axis=1))
		elif mode=='dark':
			target_label = numpy.argmin(kmeans.cluster_centers_.mean(axis=1))
	else:
		kmeans = KMeans(n_clusters=2).fit(cropped.reshape(-1,1))
		print('%.2f (%d) %.2f (%d)'%(kmeans.cluster_centers_[0], numpy.sum(kmeans.labels_==0), kmeans.cluster_centers_[1], numpy.sum(kmeans.labels_==1)))
		if mode=='light':
			target_label = numpy.argmax(kmeans.cluster_centers_)
		elif mode=='dark':
			target_label = numpy.argmin(kmeans.cluster_centers_)
#	target_label = kmeans.labels_.reshape(cropped.shape)[y-yl, x-xl]
	mask = kmeans.labels_.reshape(cropped.shape[0], cropped.shape[1])==target_label
	ym, xm = numpy.nonzero(mask)
	ym += yl
	xm += xl
	image_display[ym,xm,:] = [255,0,0]
	segment_id += 1 
	labels[ym,xm] = segment_id
	plt.clf()
	plt.imshow(image_display)
	print('Image #%d Segment #%d'%(img_id, segment_id))
	plt.title('Image #%d Segment #%d'%(img_id, segment_id))
	fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)
next_img()
plt.show()
