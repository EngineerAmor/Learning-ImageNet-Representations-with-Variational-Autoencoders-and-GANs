T12 dataset splits

train/: 11,388 images in 10 class folders.
val/: 1,265 images in 10 class folders (10% of each class, rounded).
test/: EMPTY placeholder class folders. Official TEST is pending.

Original images in ../T12_Images were copied without modification.
TRAIN and VAL match the existing project split, seed 473.
train_index.json/csv, val_index.json/csv and test_index.json/csv list the images.
Image paths in the indices are relative to this T12_Splits folder.
train_label is the contiguous 0-9 label; folder_label is the source folder prefix.
metadata.json records the protocol and status; split_counts.csv has per-class counts.

To complete TEST, supply official ILSVRC-2012 validation images and the official
validation ground truth with the devkit class-ID-to-WNID mapping. Select all 50
official validation images for each of the 10 WNIDs (500 total), map to train_label,
and update the test indices and metadata. No training images were used as TEST.
