# Video-based-human-activity-recognition

## Requirements.txt
Install the depencies by running:
```
bash Requirements.txt
```

#### Directory Structure
The important parts of the code are organized as follows.

```
main.py                          # main script (maybe parse arguments here too)
Requirements.py                  # file with the dependencies

data_utils
├── data_loader.py               # loader for the something-something v2 dataset
├── preprocess.py                #
├── metadata_loader.py           #  
├── somethingsomethingv2         # default parser
├── video_loader.py              # loader for the videos
└── default
    ├── category.txt             # file with all categories as rows
    └── someth..thingv2.py       # default parser of the dataset

training 
└── train.py                     # called to start training

data 
└── 20bn-something-something-v2  # the full dataset

models
└── Xpection.py                  # Xcpection based model
```


## Running
In main.py we keep the progresss towards the endgoal. All implementations take place outside, either in data_utils, models or training.

