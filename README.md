# d3
A High-fidelity Dynamic Driving Simulation Dataset for Stereo-depth and Sceneflow

![](assets/d3_sample.jpg)

With a focus on various weather conditions

<p float="left">
  <img src="assets/v9_0005_000130.jpg" width="200" />
  <img src="assets/v2_0009_000212.jpg" width="200" /> 
  <img src="assets/v3_0013_000366.jpg" width="200" /> 
</p>
<p float="left">
  <img src="assets/v4_0008_000166.jpg" width="200" />
  <img src="assets/v5_0006_000130.jpg" width="200" /> 
  <img src="assets/v6_0009_000366.jpg" width="200" /> 
</p>
<p float="left">
  <img src="assets/v7_0010_000474.jpg" width="200" />
  <img src="assets/v8_0011_000360.jpg" width="200" /> 
  <img src="assets/v10_0009_000240.jpg" width="200" /> 
</p>

## Download the dataset
Download the datatset form harvard datavers (link coming soon) and extract the files. Your directory should look something like this

```
- d3
    - intrinsics.json
    - test
        - 0001
            - left
            - right
            - meta
        - 0002
        - 0003
    - train
        - 0001
            - depth
            - flow
            - left
            - right
            - z_flow
            - meta
        - 0002
        - 0003
    
```

## Run the sample script
To run the script, pass in your d3 path as an argument like this 
```
python sample_script.py --root_dir /path/to/d3
```

This script iterates through the training dataset and computes our metrics on the prediction against the ground truth

