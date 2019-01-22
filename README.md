### HeadPose Forensics

#### Environment

- Ubuntu 16.04
- tqdm 4.28.1
- numpy 1.15.4
- dlib 19.16.0
- opencv-python 3.4.3.18

#### Demo

```
python run_headpose_forensic_v2.py --input_dir=debug_data
```

This will examine all images and videos in the folder of 'debug_data', print results in terminal, and also saved as "proba_list.p" in the project root folder.

I used the virtualenv as environment, and the list of packages I used is in requirements.txt.