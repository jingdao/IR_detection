import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
slim = tf.contrib.slim
import h5py
import network
import argparse
import albumentations as A
import matplotlib.pyplot as plt
import sys
import cv2

parser = argparse.ArgumentParser()
envarg = parser.add_argument_group('Training params')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--number_of_classes", type=int, default=2, help="Number of classes to be predicted.")
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=1e-3, help="initial learning rate.")
envarg.add_argument("--multi_grid", type=list, default=[1,2,4], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=4, help="Spatial Pyramid Pooling rates")
envarg.add_argument("--gpu_id", type=int, default=0, help="Id of the GPU to be used")
envarg.add_argument("--crop_width", type=int, default=385, help="Image Crop Width.")
envarg.add_argument("--crop_height", type=int, default=385, help="Image Crop Height.")
envarg.add_argument("--resnet_model", default="resnet_v2_0", choices=["resnet_v2_0", "resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"], help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")
envarg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
envarg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")
trainarg = parser.add_argument_group('Training')
trainarg.add_argument("--batch_size", type=int, default=5, help="Batch size for network train.")
trainarg.add_argument("--max_epochs", type=int, default=150, help="Max epochs for network train.")
trainarg.add_argument("--synth_prob", type=float, default=0.0, help="Probability of synthesizing additional foreground objects")
trainarg.add_argument("--dataset", type=str, default="beach", help="Dataset directory name")
trainarg.add_argument("--use_history", help="use historical data input", action="store_true")
trainarg.add_argument("--use_original", help="use original network architecture", action="store_true")
trainarg.add_argument("--viz_augmentation", help="visualize data augmentation results", action="store_true")
args = parser.parse_args()

if args.use_history:
	f = h5py.File('dataset/%s/data_history.h5'%args.dataset,'r')
else:
	f = h5py.File('dataset/%s/data.h5'%args.dataset,'r')
if args.use_original:
	args.resnet_model = "resnet_v2_50"
	args.output_stride = 16
train_img = f['train_img'][:].astype(numpy.uint8)
train_labels = f['train_labels'][:].astype(numpy.int32)
test_img = f['test_img'][:].astype(numpy.uint8)
test_labels = f['test_labels'][:].astype(numpy.int32)
f.close()

print('train',train_img.shape)
print('test',test_img.shape)
imchannels = train_img.shape[-1]

def synthesize_object(image, mask):
    new_image = image.copy()
    new_mask = mask.copy()
    ret, com = cv2.connectedComponents(mask.astype(numpy.uint8))
    for i in range(1, com.max()):
        my, mx = numpy.nonzero(com == i)
        dy = numpy.random.randint(max(-my.min(), -50), min(image.shape[0] - my.max(), 50))
        dx = numpy.random.randint(max(-mx.min(), -50), min(image.shape[1] - mx.max(), 50))
        my += dy
        mx += dx
        new_image[my, mx] = image[com == i]
        new_mask[my, mx] = mask[com == i]
#    plt.subplot(2,2,1)
#    plt.imshow(image)
#    plt.subplot(2,2,2)
#    plt.imshow(mask>0, cmap='gray')
#    plt.subplot(2,2,3)
#    plt.imshow(new_image)
#    plt.subplot(2,2,4)
#    plt.imshow(new_mask>0, cmap='gray')
#    plt.show()
    return new_image, new_mask

class MyNet:
	def __init__(self):
		self.is_training_pl = tf.placeholder(tf.bool, shape=[])
		self.input_pl = tf.placeholder(tf.float32, shape=[args.batch_size,args.crop_height,args.crop_width,imchannels])
		self.label_pl = tf.placeholder(tf.int32, shape=[args.batch_size,args.crop_height,args.crop_width])
		logits_tf = tf.cond(self.is_training_pl, true_fn= lambda: network.deeplab_v3(self.input_pl, args, is_training=True, reuse=False),
                    false_fn=lambda: network.deeplab_v3(self.input_pl, args, is_training=False, reuse=True))

		val_tp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(logits_tf, axis=-1), 1), tf.math.equal(self.label_pl, 1)),tf.int32))
		val_fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(logits_tf, axis=-1), 1), tf.math.equal(self.label_pl, 0)),tf.int32))
		val_fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(tf.argmax(logits_tf, axis=-1), 0), tf.math.equal(self.label_pl, 1)),tf.int32))
		self.val_precision = tf.cast(val_tp,tf.float32) / tf.cast(val_tp + val_fp + 1, tf.float32)
		self.val_recall = tf.cast(val_tp,tf.float32) / tf.cast(val_tp + val_fn + 1, tf.float32)

		logits_reshaped = tf.reshape(logits_tf, (args.batch_size*args.crop_height*args.crop_width,2))
		labels_reshaped = tf.reshape(self.label_pl, [args.batch_size*args.crop_height*args.crop_width])
		pos_mask = tf.where(tf.cast(labels_reshaped, tf.bool))
		neg_mask = tf.where(tf.cast(1 - labels_reshaped, tf.bool))
		self.pos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(logits_reshaped, pos_mask), labels=tf.gather_nd(labels_reshaped, pos_mask)))
		self.neg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(logits_reshaped, neg_mask), labels=tf.gather_nd(labels_reshaped, neg_mask)))
		if args.use_original:
			self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_reshaped, labels=labels_reshaped))
		else:
			self.loss = self.pos_loss + self.neg_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(args.starting_learning_rate)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
net = MyNet()
#variables_to_restore = slim.get_variables_to_restore(exclude=[args.resnet_model + "/logits", "optimizer_vars",
#                                                              "DeepLab_v3/ASPP_layer", "DeepLab_v3/logits"])
#restorer = tf.train.Saver(variables_to_restore)
#restorer.restore(sess, "./resnet/checkpoints/" + args.resnet_model + ".ckpt")
#print("Model checkpoints for " + args.resnet_model + " restored!")

# define data augmentations
transform = A.Compose([
    A.RandomCrop(width=args.crop_width, height=args.crop_height),
#    A.HorizontalFlip(p=0.5),
#    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])

saver = tf.train.Saver()
if args.use_original:
	MODEL_PATH = 'dataset/%s/original_model.ckpt' % args.dataset
elif args.use_history:
	print('use_history')
	MODEL_PATH = 'dataset/%s/history/model.ckpt' % args.dataset
else:
	MODEL_PATH = 'dataset/%s/model.ckpt' % args.dataset

init = tf.global_variables_initializer()
sess.run(init, {})
for epoch in range(args.max_epochs):
    idx = numpy.arange(len(train_labels))
    numpy.random.shuffle(idx)
    input_points = numpy.zeros((args.batch_size, args.crop_height, args.crop_width, 3))

    loss_arr = []
    pl_arr = []
    nl_arr = []
    prc_arr = []
    rcl_arr = []
    num_batches = int(len(train_labels) / args.batch_size)
    for batch_id in range(num_batches):
#		print('batch %d/%d'%(batch_id,num_batches))
        start_idx = batch_id * args.batch_size
        end_idx = (batch_id + 1) * args.batch_size
        input_images = train_img[idx[start_idx:end_idx], :, :, :]
        input_labels = train_labels[idx[start_idx:end_idx], :, :]
        augmented_images = numpy.zeros((args.batch_size, args.crop_height, args.crop_width, 3), dtype=numpy.float32)
        augmented_labels = numpy.zeros((args.batch_size, args.crop_height, args.crop_width), dtype=numpy.int32)
        for i in range(args.batch_size):
            augmented = transform(image=input_images[i], mask=input_labels[i])
            augmented_images[i] = augmented['image']
            augmented_labels[i] = augmented['mask']
            if numpy.random.random() < args.synth_prob:
                augmented_images[i], augmented_labels[i] = synthesize_object(augmented_images[i], augmented_labels[i])
            if args.viz_augmentation:
                fontsize = 18
                f, ax = plt.subplots(2, 2, figsize=(8, 8))
                ax[0, 0].imshow(input_images[i])
                ax[0, 0].set_title('Original image', fontsize=fontsize)
                ax[1, 0].imshow(input_labels[i])
                ax[1, 0].set_title('Original mask', fontsize=fontsize)
                ax[0, 1].imshow(augmented_images[i].astype(numpy.uint8))
                ax[0, 1].set_title('Transformed image', fontsize=fontsize)
                ax[1, 1].imshow(augmented_labels[i])
                ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
                plt.show()
        _, ls, pl, nl, prc, rcl = sess.run([net.train_op, net.loss, net.pos_loss, net.neg_loss, net.val_precision, net.val_recall], {net.input_pl:augmented_images, net.label_pl:augmented_labels, net.is_training_pl:True})
        loss_arr.append(ls)
        pl_arr.append(pl)
        nl_arr.append(nl)
        prc_arr.append(prc)
        rcl_arr.append(rcl)
    print("Epoch %d loss %.2f(%.2f+%.2f) prc %.2f rcl %.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(pl_arr),numpy.mean(nl_arr),numpy.mean(prc_arr),numpy.mean(rcl_arr)))

saver.save(sess, MODEL_PATH)
