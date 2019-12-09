import numpy
import scipy.misc
import h5py
import glob
import sys

dataset = 'beach'
use_history = False
for i in range(len(sys.argv)):
	if sys.argv[i]=='--dataset':
		dataset = sys.argv[i+1]
	if sys.argv[i]=='--use_history':
		use_history = True

num_samples = len(glob.glob('dataset/%s/label*.png' % dataset))
train_samples = set(range(int(0.8*num_samples)))
test_samples = set(range(num_samples)) - train_samples
if dataset == 'combined':
	train_samples -= set([0,96])
	test_samples -= set([224,248])
else:
	train_samples -= set([0])
print('train',train_samples)
print('test',test_samples)
xbound, ybound, imscale = [int(t) for t in open('dataset/%s/params.txt'%dataset).readline().split()]
pos_idx = set()
imsize = 385
train_img = numpy.zeros((len(train_samples), imsize, imsize, 3), dtype=numpy.uint8)
train_labels = numpy.zeros((len(train_samples), imsize, imsize), dtype=numpy.uint8)
test_img = numpy.zeros((len(test_samples), imsize, imsize, 3), dtype=numpy.uint8)
test_labels = numpy.zeros((len(test_samples), imsize, imsize), dtype=numpy.uint8)
train_count = 0
test_count = 0
previous_img = None

for i in range(1,num_samples+1):
	image_np = scipy.misc.imread('dataset/%s/%d.png'%(dataset,i),mode='RGB')
	image_np = image_np[ybound:ybound+imscale,xbound:xbound+imscale].mean(axis=2)
	if previous_img is None:
		diff_img = numpy.zeros(image_np.shape, dtype=numpy.uint8)
		backSub_img = numpy.zeros(image_np.shape, dtype=numpy.uint8)
	else:
		diff_img = ((image_np - previous_img)/2 + 128).astype(numpy.uint8)
		if use_history:
			backSub_img = scipy.misc.imread('dataset/%s/backSub/%d.png'%(dataset,i),mode='RGB').mean(axis=2)
	previous_img = image_np
	image_h = image_np.shape[0]
	image_w = image_np.shape[1]
	if use_history:
		image_np = numpy.dstack((image_np, diff_img, backSub_img)).astype(numpy.uint8)
	else:
		image_np = numpy.dstack((image_np, image_np, image_np)).astype(numpy.uint8)
	annotation = scipy.misc.imread('dataset/%s/label%d.png'%(dataset,i), mode='L')
	for p in numpy.array(numpy.nonzero(annotation)).T:
		pos_idx.add(tuple(p))
	annotation = annotation[ybound:ybound+imscale,xbound:xbound+imscale]
	if imscale!=imsize:
		image_np = scipy.misc.imresize(image_np, size=(imsize, imsize), interp='bilinear')
		annotation = scipy.misc.imresize(annotation, size=(imsize, imsize), interp='bilinear')
	annotation = numpy.array(annotation > 0, dtype=numpy.uint8)
	print(i,image_np.shape,image_np.dtype)
	if i-1 in train_samples:
		train_img[train_count] = image_np
		train_labels[train_count] = annotation
		train_count += 1
	elif i-1 in test_samples:
		test_img[test_count] = image_np
		test_labels[test_count] = annotation
		test_count += 1

pos_idx = numpy.array(list(pos_idx))
print('pos_idx(%d) x:%d->%d y:%d->%d'%(len(pos_idx),pos_idx[:,1].min(),pos_idx[:,1].max(),pos_idx[:,0].min(),pos_idx[:,0].max()))
print('train_count: %d test_count: %d'%(train_count, test_count))

if use_history:
	f = h5py.File('dataset/%s/data_history.h5'%dataset,'w')
else:
	f = h5py.File('dataset/%s/data.h5'%dataset,'w')
f.create_dataset('train_img',data=train_img, compression='gzip', compression_opts=4, dtype=numpy.uint8)
f.create_dataset('train_labels',data=train_labels, compression='gzip', compression_opts=4, dtype=numpy.uint8)
f.create_dataset('test_img',data=test_img, compression='gzip', compression_opts=4, dtype=numpy.uint8)
f.create_dataset('test_labels',data=test_labels, compression='gzip', compression_opts=4, dtype=numpy.uint8)
f.close()
