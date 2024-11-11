import os

import numpy as np
import torch

from backbones.BackboneBuilder import get_backbone
from dataset.transforms import *
from dataset.V1CircularScrewDataset import V1CircularScrewDataset


def cluster_dataset_split():

    TRAINV1 = ['Ankle01', 'Ankle02', 'Ankle03', 'Ankle05', 'Ankle06', 'Ankle07', 'Ankle08', 'Ankle09',
               'Ankle10', 'Ankle11', 'Ankle12', 'Ankle13', 'Ankle14', 'Ankle15', 'Ankle16', 'Ankle17', 'Elbow01',
               'Elbow02', 'Elbow03', 'Foot01', 'Knee01', 'Knee02', 'Knee03', 'Knee04', 'Knee05', 'Knee06', 'Knee07',
               'Knee08', 'Knee09', 'Leg01', 'Spine01', 'Spine02', 'Spine03', 'Spine04', 'Spine05', 'Wrist01', 'Wrist02',
               'Wrist03', 'Wrist04', 'Wrist05', 'Wrist06', 'Wrist07', ]

    VALV1 = ['Ankle21', 'Ankle22', 'Ankle23', 'Elbow04', 'Wrist11', 'Wrist12', 'Wrist13', 'Spine06', 'Spine07', ]
    
    # TESTV1 = ['Ankle18', 'Ankle19', 'Ankle20', 'Wrist08', 'Wrist09', 'Wrist10', ] # Original code
    
    # 'Ankle18' and 'Ankle19' yield NaN values ...
    # TESTV1 = ['Ankle19', 'Wrist08', 'Wrist09', 'Wrist10'] # Adapted code

    TESTV1 = ['Ankle22', 'Ankle23', 'Elbow04', 'Wrist11', 'Wrist12', 'Wrist13', 'Spine06', 'Spine07', ] # Adapted code

    print(f"Volumes for training: {TRAINV1}")
    print(f"Volumes for validation: {VALV1}")
    print(f"Volumes for testing: {TESTV1}")

    train_dirs = []
    val_dirs = []
    test_dirs = []
    for i in range(1, 4):
        train_dirs = train_dirs + [f"{element}_le_512x512x512_{i}" for element in TRAINV1]
        val_dirs = val_dirs + [f"{element}_le_512x512x512_{i}" for element in VALV1]
        test_dirs = test_dirs + [f"{element}_le_512x512x512_{i}" for element in TESTV1]

    return train_dirs, val_dirs, test_dirs


# def cluster_dataset_split():

#     # TRAINV1 = ['Ankle01', 'Ankle02', 'Ankle03', 'Ankle05', 'Ankle06', 'Ankle07', 'Ankle08', 'Ankle09',
#     #            'Ankle10', 'Ankle11', 'Ankle12', 'Ankle13', 'Ankle14', 'Ankle15', 'Ankle16', 'Ankle17', 'Elbow01',
#     #            'Elbow02', 'Elbow03', 'Foot01', 'Knee01', 'Knee02', 'Knee03', 'Knee04', 'Knee05', 'Knee06', 'Knee07',
#     #            'Knee08', 'Knee09', 'Leg01', 'Spine01', 'Spine02', 'Spine03', 'Spine04', 'Spine05', 'Wrist01', 'Wrist02',
#     #            'Wrist03', 'Wrist04', 'Wrist05', 'Wrist06', 'Wrist07', ]

#     " The following training set causes a bug "
#     # TRAINV1 = ['Ankle01', 'Ankle02', 'Ankle03', 'Ankle05', 'Ankle06', 'Ankle07', 'Ankle08', 'Ankle09',
#     #            'Ankle10', 'Ankle11', 'Ankle12', 'Ankle13', 'Ankle14', 'Ankle15', 'Ankle16', 'Ankle17', 'Elbow01',
#     #            'Elbow02', 'Elbow03', 'Knee01', 'Knee02', 'Knee03', 'Knee04', 'Knee05', 'Knee06', 'Knee07',
#     #            'Knee08', 'Knee09', 'Leg01', 'Spine01', 'Spine02', 'Spine03', 'Spine04', 'Spine05', 'Wrist01', 'Wrist02',
#     #            'Wrist03', 'Wrist04', 'Wrist05', 'Wrist06', 'Wrist07'] # Foot01_3 missing ...

#     " This training set should fix the bug "
#     TRAINV1 = ['Ankle03', 'Ankle05', 'Ankle06', 'Ankle07', 'Ankle08', 'Ankle09',
#                'Ankle10', 'Ankle11', 'Ankle12', 'Ankle13', 'Ankle14', 'Ankle15', 'Ankle16', 'Ankle17', 'Elbow01',
#                'Elbow02', 'Elbow03', 'Knee01', 'Knee02', 'Knee03', 'Knee04', 'Knee05', 'Knee06', 'Knee07',
#                'Knee08', 'Knee09', 'Leg01', 'Spine01', 'Spine02', 'Spine03', 'Spine04', 'Spine05', 'Wrist01', 'Wrist02',
#                'Wrist03', 'Wrist04', 'Wrist05', 'Wrist06', 'Wrist07'] # Ankle01, Ankle02, and Foot01_3 missing ...

#     # VALV1 = ['Ankle21', 'Ankle22', 'Ankle23', 'Elbow04', 'Wrist11', 'Wrist12', 'Wrist13', 'Spine06', 'Spine07', ]
#     VALV1 = ['Ankle21', 'Ankle22', 'Ankle23', 'Wrist11', 'Wrist12', 'Wrist13', 'Spine06', 'Spine07'] # Elbow04_3 missing ...
    
#     # TESTV1 = ['Ankle18', 'Ankle19', 'Ankle20', 'Wrist08', 'Wrist09', 'Wrist10', ]
#     TESTV1 = ['Ankle18', 'Ankle19', 'Wrist08', 'Wrist09', 'Wrist10'] # Ankle20_3 missing ...

#     print(f"\nVolumes for training: {TRAINV1}")
#     print(f"Volumes for validation: {VALV1}")
#     print(f"Volumes for testing: {TESTV1}")

#     train_dirs = []
#     val_dirs = []
#     test_dirs = []

#     for i in range(1, 4):

#         train_dirs = train_dirs + [f"{element}_{i}" for element in TRAINV1]
#         val_dirs = val_dirs + [f"{element}_{i}" for element in VALV1]
#         test_dirs = test_dirs + [f"{element}_{i}" for element in TESTV1]

#         # train_dirs = train_dirs + [f"{element}_le_512x512x512_{i}" for element in TRAINV1]
#         # val_dirs = val_dirs + [f"{element}_le_512x512x512_{i}" for element in VALV1]
#         # test_dirs = test_dirs + [f"{element}_le_512x512x512_{i}" for element in TESTV1]

#     return train_dirs, val_dirs, test_dirs


def cluster_test_dataset_split():
    TRAINV1 = [
        'Ankle01_le_512x512x512_1', 'Ankle02_le_512x512x512_1', 'Ankle03_le_512x512x512_1', 'Ankle05_le_512x512x512_1', 'Ankle06_le_512x512x512_1',
        'Ankle01_le_512x512x512_2', 'Ankle02_le_512x512x512_2', 'Ankle03_le_512x512x512_2', 'Ankle05_le_512x512x512_2', 'Ankle06_le_512x512x512_2',
        'Ankle01_le_512x512x512_3', 'Ankle02_le_512x512x512_3', 'Ankle03_le_512x512x512_3', 'Ankle05_le_512x512x512_3', 'Ankle06_le_512x512x512_3'
    ]
    VALV1 = [
        'Ankle07_le_512x512x512_1', 'Ankle08_le_512x512x512_1',
        'Ankle07_le_512x512x512_2', 'Ankle08_le_512x512x512_2',
        'Ankle07_le_512x512x512_3', 'Ankle08_le_512x512x512_3'
    ]
    TESTV1 = [
        'Ankle09_le_512x512x512_1',
        'Ankle09_le_512x512x512_2',
        'Ankle09_le_512x512x512_3'
    ]

    # TRAINV1 = ['Leg01_le_512x512x512_1' ]
    # VALV1 = ['Leg01_le_512x512x512_2']
    # TESTV1 = ['Leg01_le_512x512x512_3']

    print(f"volumes for training: {TRAINV1}")
    print(f"volumes for validation: {VALV1}")
    print(f"volumes for testing: {TESTV1}")

    return TRAINV1, VALV1, TESTV1


def local_dataset_split():
    TRAINV1 = ['Spine36']
    VALV1 = ['Spine36']
    TESTV1 = ['Spine36']

    print(f"volumes for training: {TRAINV1}")
    print(f"volumes for validation: {VALV1}")
    print(f"volumes for testing: {TESTV1}")

    return TRAINV1, VALV1, TESTV1


def custom_collate_fn(batch):

    # Separate the batch into individual components
    projections = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    indices = [item[2] for item in batch]

    # Use default collation for embedding_tensors, encoding_tensors, and indices
    if isinstance(projections[0], list):
        projections = sum(projections, [])
    projections_array = np.stack(projections)
    indices = torch.tensor(indices)

    # Custom collation for targets: keep them as a list of dictionaries
    # No padding, just return the list as is
    if isinstance(targets[0], list):
        targets = sum(targets, [])

    collated_targets = targets

    return projections_array, collated_targets, indices


def build_dataset(image_set, args):

    if os.name == 'nt':
        print("\nUsing 'local_dataset_split()' to build dataset ...")
        training_volumes, validation_volumes, testing_volumes = local_dataset_split()
        images_per_volume = 400
    else:
        print("\nUsing 'cluster_dataset_split()' to build dataset ...")
        training_volumes, validation_volumes, testing_volumes = cluster_dataset_split()
        images_per_volume = 400

    # only change this order if changing transforms.py also
    composed_transform = V1Compose([
        V1RandomFlip(horizontal=True),
        V1RandomFlip(horizontal=False),
        V1RandomRotation(),
        V1RandomSizeCrop(512,
                         976,
                         get_backbone(args)),
    ])

    if image_set == 'train':
        volumes = training_volumes
    elif image_set == 'val':
        volumes = validation_volumes
    elif image_set == 'test':
        volumes = testing_volumes
        composed_transform = V1Compose([
            V1RandomFlip(horizontal=True, p=0.),
            V1RandomFlip(horizontal=False, p=0.),
            V1RandomRotation(multiplier=0),
            V1RandomSizeCrop(976,
                             976,
                             get_backbone(args)),
        ])

    dataset = V1CircularScrewDataset(data_dir=args.data_dir,
                                     volume_names=volumes,
                                     images_per_volume=images_per_volume,
                                     neglog_normalize=args.neglog_normalize,
                                     transform=composed_transform,
                                     aux_view=args.alpha_correspondence,
                                     pre_load_labels=True,
                                     visualize=False,
                                     reduction_factor=args.dataset_reduction)

    return dataset
