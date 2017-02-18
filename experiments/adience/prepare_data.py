import pandas as pd
import pickle as pkl
import os
from scipy import misc
import numpy as np
import h5py

base_path = '/home/jos/datasets/aligned'

#folds = [h5py.File(base_path + "/fold{}.hdf5", "w")]
#dsets = [f.create_dataset("train", ())]

def load(idx):
    df = pd.read_csv('/home/jos/datasets/aligned/fold_frontal_{}_data.txt'.format(idx), sep='\t').dropna()
    df = df[df['gender'] != 'u']

    images = []
    labels = []

    for i, row in df.iterrows():
        print("Image index:", i)

        path = os.path.join(base_path, row['user_id'], 'landmark_aligned_face.{}.{}'.format(
            row['face_id'], row['original_image']
        ))
        gender = row['gender']
        if not os.path.exists(path):
            continue

        images.append(misc.imresize(misc.imread(path), (256, 256)))
        labels.append(np.asarray([1, 0], dtype='float') if gender == 'm' else np.asarray([0, 1], dtype='float'))

    images = np.asarray(images)
    labels = np.asarray(labels)

    return images, labels

all_images = []
all_labels = []

for i in range(5):
    images, labels = load(i)
    print(images.shape)
    all_images.append(images)
    all_labels.append(labels)

for i in range(5):
    with h5py.File(base_path + "/fold{}.hdf5".format(i), "w") as f:
        train_sets = list(range(5))
        train_sets.remove(i)

        f.create_dataset("train/images", data=np.concatenate([all_images[j] for j in train_sets]))
        f.create_dataset("train/labels", data=np.concatenate([all_labels[j] for j in train_sets]))

        f.create_dataset("test/images", data=all_images[i][:, 14:14+227, 14:14+227, :])
        f.create_dataset("test/labels", data=all_labels[i])
