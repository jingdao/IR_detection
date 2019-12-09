import matplotlib.pyplot as plt
import numpy
import scipy.misc
from sklearn.cluster import KMeans
import sys

dataset = 'beach'
margin = 10
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--dataset':
		dataset = sys.argv[i+1]
	elif sys.argv[i]=='--margin':
		margin = int(sys.argv[i+1])

img_id=1
#fig = plt.figure()
fig = plt.figure(figsize=(20,30))

def next_img():
	global image_np, image_display, labels, img_id, segment_id
	try:
		image_np = scipy.misc.imread('dataset/%s/%d.png'%(dataset,img_id))
	except IOError:
		sys.exit(0)
	image_display = image_np.copy()
	image_np = numpy.mean(image_np, axis=2)
	#print(image_np.shape, image_display.shape)
	labels = numpy.zeros(image_np.shape, dtype=int)
	segment_id=0
	plt.imshow(image_display)
	print('Image #%d Segment #%d'%(img_id, segment_id))
	plt.title('Image #%d Segment #%d'%(img_id, segment_id))
	fig.canvas.draw()

def onkey(event):
	if event.key==' ':
		global img_id
		scipy.misc.imsave('dataset/%s/label%d.png'%(dataset,img_id), labels)
		img_id += 1
		next_img()
	elif event.key=='r':
		next_img()

def onclick(event):
	global segment_id
	x,y = int(numpy.round(event.xdata)), int(numpy.round(event.ydata))
	xl = max(0,x-margin)
	xr = min(image_np.shape[1],x+margin)
	yl = max(0,y-margin)
	yr = min(image_np.shape[0],y+margin)
	cropped = image_np[yl:yr, xl:xr]
	kmeans = KMeans(n_clusters=2).fit(cropped.reshape(-1,1))
	print('%.2f (%d) %.2f (%d)'%(kmeans.cluster_centers_[0], numpy.sum(kmeans.labels_==0), kmeans.cluster_centers_[1], numpy.sum(kmeans.labels_==1)))
	target_label = numpy.argmax(kmeans.cluster_centers_)
#	target_label = kmeans.labels_.reshape(cropped.shape)[y-yl, x-xl]
	mask = kmeans.labels_.reshape(cropped.shape)==target_label
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
