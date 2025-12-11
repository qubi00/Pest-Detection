# Pest-Detection
## Training
Download the dataset at: https://universe.roboflow.com/mosquitos-u6ipx/mosquito-detection-dataset
Modify the filepath of train_model.py to the yaml filepath

## Inference
Once you've saved the model, you can pass it into run_model.py in order to test it live,
or run_model_recording.py to test it on a file. If you have a GPU available, use cuda.py instead.