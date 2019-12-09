# CNN-based Person Detection using Infrared Images for Night-time Intrusion Warning Systems

Supplemental material (source code and data) for the **Sensors** journal submission *CNN-based Person Detection using Infrared Images for Night-time Intrusion Warning Systems*.

## Data preparation

To download the *university* dataset, run the following script:

```
./download_data.sh
```

To prepare a new dataset, first create a subfolder under the "dataset" folder (e.g. dataset/myfolder).
Copy the image files to this subfolder using the naming scheme 1.png 2.png 3.png ... etc.
Next, create a file "params.txt" in the subfolder with a single line containing the values \[xOffset\] \[yOffset\] \[cropSize\].
This indicates that the area of interest should start at (xOffset,yOffset) and span a window of cropSize x cropSize.

To create ground truth annotations for each image, use the "annotate.py" script.
Click on relevant segments of the image to label them. Type the "space" character for the next image and the "r" character to refresh.
The "margin" parameter controls the size of the neighboring region to perform automatic flood fill.

```
python annotate.py --dataset university --margin 50
```

## Baselines

Run the baseline algorithms and compute the resulting accuracy.
"--method" can be one of "threshold", "threshold\_adp", "backSub" or "kmeans"
Result images will be displayed if the "--viz" flag is used, otherwise only the accuracy is computed.
Use the "--save_frame" flag to save results from a specific frame to the "results" folder.

```
python baselines.py --dataset university --method threshold --viz --save_frame 100
```

## Convolutional Neural Network

The first step is to convert the image files and label files into H5 files for use in training.
If the "--use_history" flag is included, the input channels will have 3 components:
(i) infrared image intensity
(ii) difference image for 1 time step
(iii) output image from running background subtraction 
Before doing this, make sure to run "baselines.py" with "--method backSub" to save background subtraction results in the "backSub" folder.
Otherwise, only the first component is used. 

```
python process_record.py --dataset beach --use_history
```

Train the network using "train.py" (150 epochs). The trained model will be saved in "dataset/myfolder/model.ckpt".

```
python train.py --dataset beach --use_history
```

Test the network and measure the accuracy. The detection threshold (a number between 0 and 1) controls 
the confidence level above which a pixel will be considered a positive detection.
Result images will be displayed if the "--viz" flag is used, otherwise only the accuracy is computed.

```
python test.py --dataset beach --use_history --detection_threshold 0.99 --viz --save_frame 100
```

## Dependencies

1. numpy
2. scipy
3. matplotlib
4. h5py
5. tensorflow
6. [Deeplab V3](https://github.com/sthalles/deeplab_v3)

## Examples

*University* sample image

![screenshot1](results/original_98.png?raw=true)

*University* detection results

![screenshot2](results/detected_cnn_98.png?raw=true)

*Beach* sample image

![screenshot3](results/original_130.png?raw=true)

*Beach* detection results

![screenshot4](results/detected_cnn_130.png?raw=true)

*Shore* sample image

![screenshot5](results/original_500.png?raw=true)

*Shore* detection results

![screenshot6](results/detected_cnn_500.png?raw=true)

