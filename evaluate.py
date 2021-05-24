import sys
import glob
import numpy
from PIL import Image
import cv2

tp = 0
fp = 0
fn = 0
obj_tp = 0
obj_fp = 0
obj_fn = 0
min_cluster = 10
max_cluster = 1000

for folder in sys.argv[1:]:
    for testfile in glob.glob('%s/prediction*.png' % folder):
        annotation = numpy.array(Image.open(testfile.replace('prediction', 'label')))
        annotation = numpy.array(annotation > 0, dtype=numpy.uint8)
        predicted = numpy.array(Image.open(testfile))
        predicted = numpy.array(predicted > 0, dtype=numpy.uint8)
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
        num_gt = 0
        num_dt = 0
        for i in range(1, gt_com.max()+1):
            if True:
                num_gt += 1
                gt_com[gt_com==i] = num_gt
            else:
                gt_com[gt_com==i] = 0
        for i in range(1, dt_com.max()+1):
            if numpy.sum(dt_com==i) > min_cluster and numpy.sum(dt_com==i) < max_cluster:
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
        print('%s prc %.2f/%.2f rcl %.2f/%.2f'%(testfile,prc,obj_prc,rcl,obj_rcl))

P = 1.0 * tp / (tp + fp)
R = 1.0 * tp / (tp + fn)
F = 2.0 * P * R / (P + R)
oP = 1.0 * obj_tp / (obj_tp + obj_fp + 1e-6)
oR = 1.0 * obj_tp / (obj_tp + obj_fn + 1e-6)
oF = 2.0 * oP * oR / (oP + oR + 1e-6)
print('Overall Precision:%.3f/%.3f Recall:%.3f/%.3f Fscore:%.3f/%.3f'%(P, oP, R, oR, F, oF))
