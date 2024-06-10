# d3
A High-fidelity Dynamic Driving Simulation Dataset for Stereo-depth and Sceneflow

![](assets/d3_sample.jpg)

With a focus on various weather conditions



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

