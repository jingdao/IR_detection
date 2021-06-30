import numpy
from PIL import Image
import h5py
import glob
import sys
import random
import matplotlib.pyplot as plt

dataset = 'beach'
use_history = False
use_rgb = False
scale_height = None
scale_width = None
shuffle = False
test_idx = None
for i in range(len(sys.argv)):
    if sys.argv[i]=='--dataset':
        dataset = sys.argv[i+1]
    if sys.argv[i]=='--use_history':
        use_history = True
    if sys.argv[i]=='--use_rgb':
        use_rgb = True
    if sys.argv[i]=='--scale_height':
        scale_height = int(sys.argv[i+1])
    if sys.argv[i]=='--scale_width':
        scale_width = int(sys.argv[i+1])
    if sys.argv[i]=='--test_idx':
        test_idx = [int(i) for i in sys.argv[i+1].split(',')]

all_samples = []
filename_offset = 14 + len(dataset)
for i in glob.glob('dataset/%s/label*.png' % dataset):
    all_samples.append(int(i[filename_offset:-4]))
num_samples = len(all_samples)
num_train_samples = int(0.8*num_samples)
all_samples = sorted(all_samples)
if test_idx is None:
    train_samples = set(all_samples[:num_train_samples])
    test_samples = set(all_samples[num_train_samples:])
else:
    test_samples = set(test_idx)
    train_samples = set(all_samples) - test_samples
if dataset == 'combined':
    train_samples -= set([0,96])
    test_samples -= set([224,248])
print('train',train_samples)
print('test',test_samples)
try:
    xbound, ybound, imwidth, imheight = [int(t) for t in open('dataset/%s/params.txt'%dataset).readline().split()]
except ValueError:
    xbound, ybound, imscale = [int(t) for t in open('dataset/%s/params.txt'%dataset).readline().split()]
    imwidth = imheight = imscale
pos_idx = set()
if scale_height is not None and scale_width is not None:
    train_img = numpy.zeros((len(train_samples), scale_height, scale_width, 3), dtype=numpy.uint8)
    train_labels = numpy.zeros((len(train_samples), scale_height, scale_width), dtype=numpy.uint8)
    test_img = numpy.zeros((len(test_samples), scale_height, scale_width, 3), dtype=numpy.uint8)
    test_labels = numpy.zeros((len(test_samples), scale_height, scale_width), dtype=numpy.uint8)
else:
    train_img = numpy.zeros((len(train_samples), imheight, imwidth, 3), dtype=numpy.uint8)
    train_labels = numpy.zeros((len(train_samples), imheight, imwidth), dtype=numpy.uint8)
    test_img = numpy.zeros((len(test_samples), imheight, imwidth, 3), dtype=numpy.uint8)
    test_labels = numpy.zeros((len(test_samples), imheight, imwidth), dtype=numpy.uint8)
train_count = 0
test_count = 0
previous_img = None

for i in all_samples:
    image_np = numpy.array(Image.open('dataset/%s/%d.png'%(dataset,i)))
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    image_np = image_np[ybound:ybound+imheight,xbound:xbound+imwidth]
    image_gray = image_np.mean(axis=2)
    if previous_img is None:
        diff_img = numpy.zeros(image_gray.shape, dtype=numpy.uint8)
        backSub_img = numpy.zeros(image_gray.shape, dtype=numpy.uint8)
    else:
        diff_img = ((image_gray - previous_img)/2 + 128).astype(numpy.uint8)
        if use_history:
            backSub_img = numpy.array(Image.open('dataset/%s/backSub/%d.png'%(dataset,i))).mean(axis=2)
    previous_img = image_gray
    image_h = image_np.shape[0]
    image_w = image_np.shape[1]
    if use_history:
        image_np = numpy.dstack((image_gray, diff_img, backSub_img)).astype(numpy.uint8)
    elif not use_rgb:
        image_np = numpy.dstack((image_gray, image_gray, image_gray)).astype(numpy.uint8)
    annotation = numpy.array(Image.open('dataset/%s/label%d.png'%(dataset,i)))
    for p in numpy.array(numpy.nonzero(annotation)).T:
        pos_idx.add(tuple(p))
    annotation = annotation[ybound:ybound+imheight,xbound:xbound+imwidth]
    if scale_height is not None and scale_width is not None and (imwidth!=scale_width or imheight!=scale_height):
        image_np = numpy.array(Image.fromarray(image_np).resize((scale_width, scale_height), resample=Image.BILINEAR))
        annotation = numpy.array(Image.fromarray(annotation).resize((scale_width, scale_height), resample=Image.NEAREST))
    annotation = numpy.array(annotation > 0, dtype=numpy.uint8)
    print(i,image_np.shape,image_np.dtype)
    if i in train_samples:
        train_img[train_count] = image_np
        train_labels[train_count] = annotation
        train_count += 1
    elif i in test_samples:
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
