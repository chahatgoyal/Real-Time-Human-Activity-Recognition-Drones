# Real-Time-Human-Activity-Recognition-Drones
This project can be implemented in real time in order to detect human activities in the footage captured by drones, UAV surveillance.

# Introduction

This Project is based upon recognizing human activities through drone survillence, it is a pert of my undergoing reserch on UAV survillance.This is based on Real-time and multi-frame based recognition algorithms. It uses different classifier on joint estimates obtained through openpose for obtaining segmented activity like Nearest Neighbors")
+ Linear SVM
+ RBF SVM
+ Gaussian Process
+ Decision Tree
+ Random Forest
+ Custom Dataset(provided model is trained on ths classifier)
# Results

![alt text](https://github.com/chahatgoyal/Real-Time-Human-Activity-Recognition-Drones/blob/master/results/Picture6.png)
![Output sample](https://github.com/chahatgoyal/Real-Time-Human-Activity-Recognition-Drones/blob/master/results/ezgif.com-video-to-gif.gif)

# Algorithm
+ Get the joints' positions by OpenPose.
+ Track each person. Euclidean distance between the joints of two skeletons is used for matching two skeletons. See class Tracker in lib_tracker.py
+ Fill in a person's missing joints by these joints' relative pos in previous frame. See class FeatureGenerator in lib_feature_proc.py. So does the following.
+ Add noise to the (x, y) joint positions to try to augment data.
+ Use a window size of 0.5s (5 frames) to extract features.
+ Extract features of (1) body velocity and (2) normalized joint positions and (3) joint velocities.
+ Apply PCA to reduce feature dimension to 80. Classify by DNN of 3 layers of 50x50x50 (or switching to other classifiers in one line). See class neural net ClassifierOfflineTrain in lib_classifier.py
+ Mean filtering the prediction scores between 2 frames. Add label above the person if the score is larger than 0.8. See class ClassifierOnlineTest in lib_classifier.py

![alt text](https://github.com/chahatgoyal/Real-Time-Human-Activity-Recognition-Drones/blob/master/doc/joints_order.png)

The algorithm has been considered from
[report](https://github.com/felixchenfy/Data-Storage/blob/master/EECS-433-Pattern-Recognition/FeiyuChen_Report_EECS433.pdf)  


# Dataset
The dataset I have used for the given project is
[link to dataset](https://www.google.com/url?q=https://drive.google.com/file/d/1sZYRQpRpRimpSI_M5Rjjwnpc8fgI3XXM/view?usp%3Ddrive_web&source=gmail&ust=1589063688845000&usg=AFQjCNHEVQx4v_ql6ocsoRx6Y0n4Sj2Pag)

It consists of 12 actions 'Boxing','Clapping','Hitting Bottle','Hitting Stick','Jogging front back','Jogging side',
'Kicking','Running front back','Running Side','Stabbing','Walking fb','Walking Side','Waving Hands' with about 2000 images of each action.

# Execute Project
In order to execute the following project

# Install Dependencies
+ Python >= 3.6
+ Openpose: I used the OpenPose from this Github: [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)

Follow its tutorial [here](https://github.com/ildoonet/tf-pose-estimation#install-1) to download the "cmu" model. As for the "mobilenet_thin", it's already inside the folder.  

```
$ cd tf-pose-estimation/models/graph/cmu  
$ bash download.sh  
```

Then install dependencies. I listed my installation steps as bellow:
```
conda create -n tf tensorflow-gpu
conda activate tf

cd $MyRoot
pip install -r requirements.txt
pip install jupyter tqdm
pip install tensorflow-gpu==1.13.1
sudo apt install swig
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

cd $MyRoot/src/githubs/tf-pose-estimation/tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

Make sure you can successfully run its demo examples:
```
cd $MyRoot/src/githubs/tf-pose-estimation
python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```


## Run scripts
The 5 main scripts are under `src/`. They are named under the order of excecution:
```
src/s1_get_skeletons_from_training_imgs.py    
src/s2_put_skeleton_txts_to_a_single_txt.py
src/s3_preprocess_features.py
src/s4_train.py 
src/s5_test.py
```

The input and output of these files as well as some parameters are defined in the configuration file [config/config.yaml](config/config.yaml).

The script [src/s5_test.py](src/s5_test.py) is for doing real-time action recognition. 


The classes are set in [config/config.yaml](config/config.yaml) by the key `classes`.

The features obtained after running s3_preprocess_features.py for my dataset are provided below:
[FeatureX.csv](https://drive.google.com/file/d/1iAieiINnOtFTwRmunfzxDzYPkD5U0VAH/view?usp=sharing)
[FeatureY.csv:](https://drive.google.com/file/d/1RyKjH-rjpWdWUgqUjOBcpehowxZO27aL/view?usp=sharing)


The supported input includes **video file**, **a folder of images**, and **web camera**, which is set by the command line arguments `--data_type` and `--data_path`.

The trained model is set by `--model_path`, e.g.:[model/trained_classifier.pickle](model/trained_classifier.pickle).

The output is set by `--output_folder`, e.g.: output/.

The test data (images) are already included under the [data_test/](data_test/) folder.

Example commands are given below:

## Test on video file
``` bash
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type video \
    --data_path data_test/exercise.avi \
    --output_folder output
```

## Test on a folder of images
``` bash
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type folder \
    --data_path data_test/apple/ \
    --output_folder output
```

## Test on web camera
``` bash
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type webcam \
    --data_path 0 \
    --output_folder output
```

# Run on Custom Dataset

Put your data inside the folder data with images of an activity inside the folder with the same name as activity class given in config file. 

# Resuts and Performance

![alt text](https://github.com/chahatgoyal/Real-Time-Human-Activity-Recognition-Drones/blob/master/results/image1.PNG)

![alt text](https://github.com/chahatgoyal/Real-Time-Human-Activity-Recognition-Drones/blob/master/results/Picture3.jpg)



Further Extensions include exploring its perspective from lstm classifiers






