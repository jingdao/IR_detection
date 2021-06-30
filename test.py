import numpy
import matplotlib.pyplot as plt
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
slim = tf.contrib.slim
import h5py
import network
import argparse
import scipy.special
import time
import sys
import sklearn.metrics
import glob
from PIL import Image

parser = argparse.ArgumentParser()
envarg = parser.add_argument_group('Training params')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--number_of_classes", type=int, default=2, help="Number of classes to be predicted.")
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=1e-2, help="initial learning rate.")
envarg.add_argument("--multi_grid", type=list, default=[1,2,4], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=4, help="Spatial Pyramid Pooling rates")
envarg.add_argument("--gpu_id", type=int, default=0, help="Id of the GPU to be used")
envarg.add_argument("--crop_width", type=int, default=385, help="Image Crop Width.")
envarg.add_argument("--crop_height", type=int, default=385, help="Image Crop Height.")
envarg.add_argument("--resnet_model", default="resnet_v2_0", choices=["resnet_v2_0","resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"], help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")
envarg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
envarg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")
trainarg = parser.add_argument_group('Training')
trainarg.add_argument("--batch_size", type=int, default=1, help="Batch size for network train.")
testarg = parser.add_argument_group('Testing')
testarg.add_argument("--test_idx", type=str, help="Specify image indices for testing")
testarg.add_argument("--detection_threshold", type=float, default=0.5, help="Confidence threshold for detection")
testarg.add_argument("--scale_width", type=int, help="Image Scale Width")
testarg.add_argument("--scale_height", type=int, help="Image Scale Height")
testarg.add_argument("--min_cluster", type=int, default=10, help="Minimum cluster size for detection")
testarg.add_argument("--max_cluster", type=int, default=200, help="Maximum cluster size for detection")
testarg.add_argument("--use_rgb", help="use RGB color images", action="store_true")
testarg.add_argument("--dataset", type=str, default="beach", help="Dataset directory name")
testarg.add_argument("--viz", help="Visualize results", action="store_true")
testarg.add_argument("--use_history", help="use historical data input", action="store_true")
testarg.add_argument("--use_original", help="use original network architecture", action="store_true")
testarg.add_argument("--save", help="Save results", action="store_true")
testarg.add_argument("--save_frame", type=int, default=-1, help="Frame ID to save")
args = parser.parse_args()

tp = 0
fp = 0
fn = 0
obj_tp = 0
obj_fp = 0
obj_fn = 0
zoomed_in = True
fig = plt.figure(figsize=(20,30))
try:
    xbound, ybound, imwidth, imheight = [int(t) for t in open('dataset/%s/params.txt'%args.dataset).readline().split()]
except ValueError:
    xbound, ybound, imscale = [int(t) for t in open('dataset/%s/params.txt'%args.dataset).readline().split()]
    imwidth = imheight = imscale
detection_threshold = args.detection_threshold

all_samples = []
filename_offset = 14 + len(args.dataset)
for i in glob.glob('dataset/%s/label*.png' % args.dataset):
    all_samples.append(int(i[filename_offset:-4]))
num_samples = len(all_samples)
num_train_samples = int(0.8*num_samples)
all_samples = sorted(all_samples)
if args.test_idx is None:
    train_samples = set(all_samples[:num_train_samples])
    test_samples = set(all_samples[num_train_samples:])
else:
    test_samples = set([int(i) for i in args.test_idx.split(',')])
    train_samples = set(all_samples) - test_samples

print('Using detection_threshold:%.2f'%detection_threshold)

class MyNet:
    def __init__(self):
        if args.scale_height is not None and args.scale_width is not None:
            self.input_pl = tf.placeholder(tf.float32, shape=[args.batch_size,args.scale_height,args.scale_width,3])
            self.label_pl = tf.placeholder(tf.int32, shape=[args.batch_size,args.scale_height,args.scale_width])
        else:
            self.input_pl = tf.placeholder(tf.float32, shape=[args.batch_size,imheight,imwidth,3])
            self.label_pl = tf.placeholder(tf.int32, shape=[args.batch_size,imheight,imwidth])
        self.logits_tf = network.deeplab_v3(self.input_pl, args, is_training=True, reuse=False)

        self.val_tp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(self.logits_tf, axis=-1), 1), tf.math.equal(self.label_pl, 1)),tf.int32))
        self.val_fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(self.logits_tf, axis=-1), 1), tf.math.equal(self.label_pl, 0)),tf.int32))
        self.val_fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(self.logits_tf, axis=-1), 0), tf.math.equal(self.label_pl, 1)),tf.int32))
        self.val_precision = tf.cast(self.val_tp,tf.float32) / tf.cast(self.val_tp + self.val_fp + 1, tf.float32)
        self.val_recall = tf.cast(self.val_tp,tf.float32) / tf.cast(self.val_tp + self.val_fn + 1, tf.float32)

        if args.scale_height is not None and args.scale_width is not None:
            logits_reshaped = tf.reshape(self.logits_tf, (args.batch_size*args.scale_height*args.scale_width,2))
            labels_reshaped = tf.reshape(self.label_pl, [args.batch_size*args.scale_height*args.scale_width])
        else:
            logits_reshaped = tf.reshape(self.logits_tf, (args.batch_size*imheight*imwidth,2))
            labels_reshaped = tf.reshape(self.label_pl, [args.batch_size*imheight*imwidth])
        pos_mask = tf.where(tf.cast(labels_reshaped, tf.bool))
        neg_mask = tf.where(tf.cast(1 - labels_reshaped, tf.bool))
        self.pos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(logits_reshaped, pos_mask), labels=tf.gather_nd(labels_reshaped, pos_mask)))
        self.neg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(logits_reshaped, neg_mask), labels=tf.gather_nd(labels_reshaped, neg_mask)))
        if args.use_original:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_reshaped, labels=labels_reshaped))
        else:
            self.loss = self.pos_loss + self.neg_loss

if args.use_original:
	args.resnet_model = "resnet_v2_50"
	args.output_stride = 16
	MODEL_PATH = 'dataset/%s/original_model.ckpt' % args.dataset
elif args.use_history:
	MODEL_PATH = 'dataset/%s/history/model.ckpt' % args.dataset
else:
	MODEL_PATH = 'dataset/%s/model.ckpt' % args.dataset
#MODEL_PATH = 'dataset/combined/history/model.ckpt'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
net = MyNet()
saver = tf.train.Saver()
saver.restore(sess, MODEL_PATH)
print('Restored network from %s'%MODEL_PATH)

if args.scale_height is not None and args.scale_width is not None:
    input_images = numpy.zeros((args.batch_size, args.scale_height, args.scale_width, 3), dtype=numpy.float32)
    input_labels = numpy.zeros((args.batch_size, args.scale_height, args.scale_width), dtype=numpy.int32)
else:
    input_images = numpy.zeros((args.batch_size, imheight, imwidth, 3), dtype=numpy.float32)
    input_labels = numpy.zeros((args.batch_size, imheight, imwidth), dtype=numpy.int32)
img_history = []
ap_gt = []
ap_dt = []
comp_time = []
previous_img = None
for image_idx in range(1,num_samples+1):
    t1 = time.time()
    original_image = numpy.array(Image.open('dataset/%s/%d.png'%(args.dataset,image_idx)))
    if original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]
    image_np = original_image[ybound:ybound+imheight,xbound:xbound+imwidth]
    image_gray = image_np.mean(axis=2)
    if args.use_history:
        if previous_img is None:
            diff_img = numpy.zeros(image_gray.shape, dtype=numpy.uint8)
            backSub_img = numpy.zeros(image_gray.shape, dtype=numpy.uint8)
        else:
            diff_img = ((image_gray - previous_img)/2 + 128).astype(numpy.uint8)
            backSub_img = numpy.array(Image.open('dataset/%s/backSub/%d.png'%(args.dataset,image_idx))).mean(axis=2)
        previous_img = image_gray
    if not image_idx in test_samples:
        continue
    if args.use_history:
        image_np = numpy.dstack((image_gray, diff_img, backSub_img)).astype(numpy.uint8)
    elif not args.use_rgb:
        image_np = numpy.dstack((image_gray, image_gray, image_gray)).astype(numpy.uint8)
    original_annotation = numpy.array(Image.open('dataset/%s/label%d.png'%(args.dataset,image_idx)))
    annotation = original_annotation[ybound:ybound+imheight,xbound:xbound+imwidth]
    annotation = numpy.array(annotation > 0, dtype=numpy.uint8)
    if args.scale_height is not None and args.scale_width is not None and (imwidth!=args.scale_width or imheight!=args.scale_height):
        image_np = numpy.array(Image.fromarray(image_np).resize((args.scale_width, args.scale_height), resample=Image.BILINEAR))
        annotation = numpy.array(Image.fromarray(annotation).resize((args.scale_width, args.scale_height), resample=Image.NEAREST))
    input_images[:] = image_np
    input_labels[:] = annotation

    result, ls, pl, nl, prc, rcl, vtp, vfp, vfn = sess.run(
        [net.logits_tf, net.loss, net.pos_loss, net.neg_loss, net.val_precision, net.val_recall, net.val_tp, net.val_fp, net.val_fn],
        {net.input_pl:input_images, net.label_pl:input_labels})
    t2 = time.time()
#    print('Loss %.2f(%.2f+%.2f) tp:%d fp:%d fn:%d prc:%.2f rcl:%.2f'%(ls,pl,nl,vtp,vfp,vfn,prc,rcl))
    predicted_softmax = scipy.special.softmax(result[0,:,:,:],axis=-1)[:,:,1]
    ap_dt.extend(predicted_softmax.flatten())
    ap_gt.extend(annotation.flatten())
    predicted = predicted_softmax > detection_threshold
    current_tp = numpy.sum(numpy.logical_and(annotation==1, predicted==1))
    current_fp = numpy.sum(numpy.logical_and(annotation==0, predicted==1))
    current_fn = numpy.sum(numpy.logical_and(annotation==1, predicted==0))
    prc = 1.0 * current_tp / (current_tp + current_fp + 1)
    rcl = 1.0 * current_tp / (current_tp + current_fn + 1)
    tp += current_tp
    fp += current_fp
    fn += current_fn

    ret, dt_com = cv2.connectedComponents(predicted.astype(numpy.uint8))
    ret, gt_com = cv2.connectedComponents(annotation.astype(numpy.uint8))
    print('connectedComponents %d dt %d gt' % (dt_com.max(), gt_com.max()))
    num_gt = 0
    num_dt = 0
    for i in range(1, gt_com.max()+1):
#        if numpy.sum(gt_com==i) > args.min_cluster and numpy.sum(gt_com==i) < args.max_cluster:
        if True:
            num_gt += 1
            gt_com[gt_com==i] = num_gt
        else:
            gt_com[gt_com==i] = 0
    for i in range(1, dt_com.max()+1):
        if numpy.sum(dt_com==i) > args.min_cluster and numpy.sum(dt_com==i) < args.max_cluster:
            num_dt += 1
            dt_com[dt_com==i] = num_dt
        else:
            dt_com[dt_com==i] = 0
    current_tp = 0
    dt_matched = numpy.zeros(num_dt, dtype=bool)
    for i in range(1, gt_com.max()+1):
        for j in range(1, dt_com.max()+1):
            if dt_matched[j-1]:
                continue
            m1 = gt_com==i
            m2 = dt_com==j
            iou = 1.0 * numpy.sum(numpy.logical_and(m1, m2)) / numpy.sum(numpy.logical_or(m1, m2))
            if iou > 0:
                current_tp += 1
                dt_matched[j-1] = True
                break
    current_fp = numpy.sum(dt_matched==0)
    current_fn = num_gt - current_tp
    obj_tp += current_tp
    obj_fp += current_fp
    obj_fn += current_fn
    obj_prc = 0.0 if current_tp==0 else 1.0 * current_tp / (current_tp + current_fp)
    obj_rcl = 0.0 if current_tp==0 else 1.0 * current_tp / (current_tp + current_fn)

    gt_viz = numpy.zeros((gt_com.shape[0], gt_com.shape[1], 3), dtype=numpy.uint8)
    for i in range(1, gt_com.max()+1):
        c = numpy.random.randint(0,255,3)
        gt_viz[gt_com==i] = c
        my, mx = numpy.nonzero(gt_com==i)
        x1 = max(mx.min() - 5, 0)
        x2 = min(mx.max() + 5, gt_viz.shape[1] - 1)
        y1 = max(my.min() - 5, 0)
        y2 = min(my.max() + 5, gt_viz.shape[0] - 1)
        gt_viz[y1, x1:x2, :] = [255,255,0]
        gt_viz[y2, x1:x2, :] = [255,255,0]
        gt_viz[y1:y2, x1, :] = [255,255,0]
        gt_viz[y1:y2, x2, :] = [255,255,0]
    dt_viz = numpy.zeros((dt_com.shape[0], dt_com.shape[1], 3), dtype=numpy.uint8)
    for i in range(1, dt_com.max()+1):
        c = numpy.random.randint(0,255,3)
        dt_viz[dt_com==i] = c
        my, mx = numpy.nonzero(dt_com==i)
        x1 = max(mx.min() - 5, 0)
        x2 = min(mx.max() + 5, dt_viz.shape[1] - 1)
        y1 = max(my.min() - 5, 0)
        y2 = min(my.max() + 5, dt_viz.shape[0] - 1)
        dt_viz[y1, x1:x2, :] = [255,255,0]
        dt_viz[y2, x1:x2, :] = [255,255,0]
        dt_viz[y1:y2, x1, :] = [255,255,0]
        dt_viz[y1:y2, x2, :] = [255,255,0]

    comp_time.append(t2 - t1)
    print('%d/%d images prc %.2f/%.2f rcl %.2f/%.2f (%.2fs)'%(image_idx,num_samples,prc,obj_prc,rcl,obj_rcl,t2-t1))

    if image_idx == args.save_frame:
        Image.fromarray(image_np[:,:,0].astype(numpy.uint8), mode='L').save('results/original_%d.png'%args.save_frame)
        Image.fromarray(dt_viz, mode='RGB').save('results/detected_%s_%d.png'%('history' if args.use_history else 'cnn', args.save_frame))
        Image.fromarray(gt_viz, mode='RGB').save('results/ground_truth_%d.png'%args.save_frame)
        print('save_frame',args.save_frame)
        sys.exit(1)
    if args.save:
        Image.fromarray(predicted).save('dataset/%s/prediction%d.png'%(args.dataset, image_idx))
    if args.viz:
        plt.clf()
        plt.subplot(2,2,1)
        if args.use_rgb:
            plt.imshow(image_np if zoomed_in else original_image)
        else:
            plt.imshow(image_np[:,:,0] if zoomed_in else original_image, cmap='gray')
        plt.title('Image #%d'%image_idx)
        plt.subplot(2,2,2)
        plt.imshow(annotation if zoomed_in else original_annotation, cmap='gray')
        plt.subplot(2,2,3)
        plt.imshow(dt_viz)
        plt.subplot(2,2,4)
        plt.imshow(gt_viz)
        plt.show()
#        plt.pause(1.5)

AP = sklearn.metrics.average_precision_score(ap_gt, ap_dt)
P = 1.0 * tp / (tp + fp)
R = 1.0 * tp / (tp + fn)
F = 2.0 * P * R / (P + R)
oP = 1.0 * obj_tp / (obj_tp + obj_fp + 1e-6)
oR = 1.0 * obj_tp / (obj_tp + obj_fn + 1e-6)
oF = 2.0 * oP * oR / (oP + oR + 1e-6)
print('Overall %d pixels AP:%.3f Precision:%.3f/%.3f Recall:%.3f/%.3f Fscore:%.3f/%.3f (t=%.3fs)'%(len(ap_dt), AP, P, oP, R, oR, F, oF, numpy.mean(comp_time)))
