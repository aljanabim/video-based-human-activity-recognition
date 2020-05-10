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
config.py                        # file with all the parameters used across all modules

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
├──  20bn-something-something-v2-video  # the full dataset
├── 20bn-something-something-v2-jason   # jason labels comming with the dataset
├── 20bn-something-something-v2-frame   # extraced frames
└── 20bn-something-something-v2-label   # labels for train, test and valid set   

models
└── Xpection.py                  # Xcpection based model
```


## Running
In main.py we keep the progresss towards the endgoal. All implementations take place outside, either in data_utils, models or training.

# Cloud computing - GCP
## How to install GCP support on my machine?
* __Windows/Liunux/Mac:__ Follow the steps shown [here](https://cloud.google.com/sdk/docs/downloads-interactive) to get the gcloud CLI.
* __Ubuntu (apt-get):__ Follow the steps shown [here](https://cloud.google.com/sdk/docs/downloads-apt-get) to get the gcloud CLI.

It will prompt you to login once the installation is finished. The log in is done through your webb browser, make sure to use log in using the same account as the one with the 50$ on GCP. The it will ask you to type in the number corresponding to our project. If you are correctly added to them team you should see DL-DD2424 as a GCP project.

If using Powershell on Windows, make sure to restart powershell in order to use the _gcloud_ commands.

Once the installation is confirmed, run
```bash
gcloud components list # to see all you installed components
gcloud components install app-engine-python # to install the app engine support for python
```

pip install keras-tuner