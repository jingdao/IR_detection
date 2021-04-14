import matplotlib.pyplot as plt
import numpy
from PIL import Image
import sys
import cv2
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import glob
import os

method = 'threshold'
#method = 'threshold_adp'
#method = 'backSub'
#method = 'kmeans'
dataset = 'beach'
save_frame = -1 #98, 130
min_cluster = 10
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--method':
		method = sys.argv[i+1]
	elif sys.argv[i]=='--dataset':
		dataset = sys.argv[i+1]
	elif sys.argv[i]=='--save_frame':
		save_frame = int(sys.argv[i+1])
	elif sys.argv[i]=='--min_cluster':
		min_cluster = int(sys.argv[i+1])
	
backSub = cv2.createBackgroundSubtractorMOG2()
#backSub = cv2.createBackgroundSubtractorKNN()
image_id = 1
fig = plt.figure(figsize=(20,30))
try:
    xbound, ybound, imwidth, imheight = [int(t) for t in open('dataset/%s/params.txt'%dataset).readline().split()]
except ValueError:
    xbound, ybound, imscale = [int(t) for t in open('dataset/%s/params.txt'%dataset).readline().split()]
    imwidth = imheight = imscale
num_samples = len(glob.glob('dataset/%s/label*.png'%dataset))
num_test = num_samples - int(num_samples*0.8)
test_idx = num_samples - num_test + 1
tp = 0
fp = 0
fn = 0
obj_tp = 0
obj_fp = 0
obj_fn = 0
viz = '--viz' in sys.argv
zoomed_in = True
comp_time = []

while True:
	if method!='backSub' and image_id < test_idx:
		image_id += 1
		continue
	image_filename = 'dataset/%s/%d.png' % (dataset,image_id)
	label_filename = 'dataset/%s/label%d.png'%(dataset,image_id)
	if os.path.exists(image_filename) and os.path.exists(label_filename):
		I = numpy.array(Image.open(image_filename))
		if len(I.shape)>2:
			I = numpy.mean(I, axis=2)
	else:
		break
	gt = numpy.array(Image.open(label_filename))
	gt = gt > 0
	dt = numpy.zeros(I.shape, dtype=bool)
	image_np = I[ybound:ybound+imheight, xbound:xbound+imwidth]
	t1 = time.time()
	if method=='threshold':
		Isub = image_np.astype(numpy.uint8)
		val, mask = cv2.threshold(Isub,75 if dataset=='beach' else 85 if dataset=='shore' else 120,255,cv2.THRESH_BINARY)
	elif method=='threshold_adp':
		Isub = image_np.astype(numpy.uint8)
		blur = cv2.medianBlur(Isub,5)
		if dataset=='beach':
			mask = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,-5)
		elif dataset=='shore':
			mask = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,-8)
		else:
			mask = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,-10)
	elif method=='backSub':
		if dataset=='combined' and image_id in [97, 225, 249]:
			backSub = cv2.createBackgroundSubtractorMOG2()
		mask = backSub.apply(image_np)
		Image.fromarray(mask.astype(numpy.uint8), mode='L').save('dataset/%s/backSub/%d.png'%(dataset,image_id))
	elif method=='kmeans':
		window_size = 15 if dataset=='beach' or dataset=='shore' else 100
		margin = 10 if dataset=='beach' or dataset=='shore' else 100
		Isub = image_np.copy()
		#start with mean shift
		centerX = 0
		centerY = 0
		centerVal = Isub[centerY, centerX]
		peaks = []
		peakVal = []
		while True:
			while True:
				x1 = max(0,centerX-window_size)
				x2 = min(Isub.shape[1],centerX+window_size)
				y1 = max(0,centerY-window_size)
				y2 = min(Isub.shape[0],centerY+window_size)
				Itmp = Isub[y1:y2,x1:x2]
				maxVal = Itmp.max()
#				print(centerX,centerY,centerVal,maxVal)
				if maxVal > centerVal:
					dy, dx = numpy.unravel_index(numpy.argmax(Itmp), Itmp.shape)
					centerY = y1+dy
					centerX = x1+dx
					centerVal = maxVal
					Isub[y1:y2,x1:x2] = 0
				else:
					peaks.append([centerX,centerY])
					peakVal.append(centerVal)
					Isub[y1:y2,x1:x2] = 0
#					print('Found peak (%d,%d) at %d'%(centerX,centerY,centerVal))
					break
			valid_idx = numpy.array(numpy.nonzero(Isub)).T
			if len(valid_idx) > 0:
				centerY, centerX = valid_idx[0]
				centerVal = Isub[centerY, centerX]
			else:
				break
		kmeans = KMeans(n_clusters=2).fit(numpy.array(peakVal).reshape(-1,1))
#		print(kmeans.cluster_centers_, numpy.sum(kmeans.labels_==0), numpy.sum(kmeans.labels_==1))
		target_label = numpy.argmax(kmeans.cluster_centers_)
		if dataset=='beach':
			peaks = numpy.array(peaks)[numpy.array(peakVal)>100]
		elif dataset=='shore':
			peaks = numpy.array(peaks)[numpy.array(peakVal)>85]
		else:
			peaks = numpy.array(peaks)[kmeans.labels_ == target_label]
		Isub = image_np.copy()
		mask = numpy.zeros(Isub.shape, dtype=bool)
		for x,y in peaks:
			xl = max(0,x-margin)
			xr = min(Isub.shape[1],x+margin)
			yl = max(0,y-margin)
			yr = min(Isub.shape[0],y+margin)
			cropped = Isub[yl:yr, xl:xr]
			kmeans = KMeans(n_clusters=2).fit(cropped.reshape(-1,1))
#			print('kmeans %.2f (%d) %.2f (%d)'%(kmeans.cluster_centers_[0], numpy.sum(kmeans.labels_==0), kmeans.cluster_centers_[1], numpy.sum(kmeans.labels_==1)))
			target_label = numpy.argmax(kmeans.cluster_centers_)
			M = kmeans.labels_.reshape(cropped.shape)==target_label
			ym, xm = numpy.nonzero(M)
			ym += yl
			xm += xl
			mask[ym,xm] = True
	t2 = time.time()
	dt[ybound:ybound+imheight,xbound:xbound+imwidth] = mask
	err_viz = numpy.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=numpy.uint8)
	if image_id < test_idx:
		image_id += 1
		continue

	gt_sub = gt[ybound:ybound+imheight, xbound:xbound+imwidth] > 0
	dt_sub = dt[ybound:ybound+imheight, xbound:xbound+imwidth]	
	current_tp = numpy.logical_and(gt_sub,dt_sub)
	current_fp = numpy.logical_and(numpy.logical_not(gt_sub),dt_sub)
	current_fn = numpy.logical_and(gt_sub,numpy.logical_not(dt_sub))
	err_viz[current_tp] = [0,255,0]
	err_viz[current_fp] = [0,0,255]
	err_viz[current_fn] = [255,0,0]
	current_tp = numpy.sum(current_tp)
	current_fp = numpy.sum(current_fp)
	current_fn = numpy.sum(current_fn)
	prc = 1.0*current_tp/(current_tp+current_fp+1)
	rcl = 1.0*current_tp/(current_tp+current_fn+1)
	tp += current_tp
	fp += current_fp
	fn += current_fn

	ret, gt_com = cv2.connectedComponents(gt_sub.astype(numpy.uint8))
	ret, dt_com = cv2.connectedComponents(dt_sub.astype(numpy.uint8))
	num_gt = 0
	num_dt = 0
	for i in range(1, gt_com.max()+1):
		if numpy.sum(gt_com==i) > min_cluster:
			num_gt += 1
			gt_com[gt_com==i] = num_gt
		else:
			gt_com[gt_com==i] = 0
	for i in range(1, dt_com.max()+1):
		if numpy.sum(dt_com==i) > min_cluster:
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
	obj_prc = 1.0 * current_tp / (current_tp + current_fp) if current_tp > 0 else 0
	obj_rcl = 1.0 * current_tp / (current_tp + current_fn) if current_tp > 0 else 0

	gt_viz = numpy.zeros((gt_sub.shape[0], gt_sub.shape[1], 3), dtype=numpy.uint8)
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
	dt_viz = numpy.zeros((dt_sub.shape[0], dt_sub.shape[1], 3), dtype=numpy.uint8)
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
	print('Image #%d Precision:%.2f/%.2f Recall:%.2f/%.2f (%.2fs)'%(image_id, prc,obj_prc,rcl,obj_rcl, t2-t1))

	if image_id == save_frame:
		Image.fromarray(image_np.astype(numpy.uint8), mode='L').save('results/original_%d.png'%save_frame)
		Image.fromarray(dt_viz, mode='RGB').save('results/detected_%s_%d.png'%(method, save_frame))
		Image.fromarray(gt_viz, mode='RGB').save('results/ground_truth_%d.png'%save_frame)
		print('save_frame',save_frame)
		sys.exit(1)
	if viz:
		plt.clf()
		plt.subplot(2,2,1)
		plt.imshow(image_np if zoomed_in else I, cmap='gray')
		plt.title('Image #%d'%image_id)
		plt.subplot(2,2,2)
		plt.imshow(gt_sub if zoomed_in else gt, cmap='gray')
		plt.subplot(2,2,3)
		plt.imshow(dt_viz if zoomed_in else dt, cmap='gray')
		plt.subplot(2,2,4)
		plt.imshow(gt_viz, cmap='gray')
		plt.pause(0.5)
	image_id += 1

P = 1.0 * tp / (tp + fp)
R = 1.0 * tp / (tp + fn)
F = 2.0 * P * R / (P + R)
oP = 1.0 * obj_tp / (obj_tp + obj_fp)
oR = 1.0 * obj_tp / (obj_tp + obj_fn)
oF = 2.0 * oP * oR / (oP + oR)
print('Overall Precision:%.3f/%.3f Recall:%.3f/%.3f Fscore:%.3f/%.3f (t=%.6fs)'%(P, oP, R, oR, F, oF, numpy.mean(comp_time)))
