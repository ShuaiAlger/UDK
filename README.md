# UDK

### requirments
```
pytorch >= 1.8.0
numpy
opencv-python >= 4.4.0
argparse
PyYaml
matplotlib
loguru
glob
h5py
kornia
pathlib
tqdm
```
### Test Data

[HPatches](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)

Please remove the following folders after downloading: i_contruction i_crownnight i_dc i_pencils i_whitebuilding v_artisans v_astronautis v_talent.

[megadepth-1500 and scannet-1500](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) 

### Evaluation 

python dfm.py

The settings of various algorithms are recorded in configs/xxx.yml, dfm.py, and ManyDeepFeatureMatcher.py.

By the way, the ManyDeepFeatureMatcher.py mainly controls the detection strategies, detection params, and the level of multi-scale.

We will optimize the parameters structure as soon as possible.

We would like to thank the SuperGlue, LoFTR and DFM authors and contributors for making their codes or data open source which inspired.


