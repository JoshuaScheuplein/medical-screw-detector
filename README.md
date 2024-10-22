# Medical Screw Detector

## Getting started

### Building custom cuda kernels
```
python3 setup.py build
python3 setup.py install --prefix=~/.local/
```

### Environment
Check the environments for different setups:
- [Conda Environment](env/environment.yml)
- [pip Cluster Environment](env/requirements_cluster.txt)
- [pip Local Environment](env/requirements_local.txt)

### Running
```
python main.py
  --backbone resnet101
  --dataset_reduction 10
  --lr 0.00004 
  --lr_backbone 0.000004 
  --batch_size 2 
  --epochs 47 
  --lr_drop_epochs 40 
  --data_dir <path_to_dataset>
  --result_dir <path_for_results>
  --checkpoint_file <path_to_checkpoint>.ckpt
  --backbone_checkpoint_file <path_to_backbone_checkpoint>
```

### ResNet50 Test
```
srun python main.py --backbone medical_resnet50 --dataset_reduction 10 --lr 0.00004 --lr_backbone 0.000004 --batch_size 2 --epochs 5 --lr_drop_epochs 2 --data_dir /home/vault/iwi5/iwi5165h/2024-04-Scheuplein-Screw-Detection --result_dir /home/hpc/iwi5/iwi5165h/Screw-Detection-Results --backbone_checkpoint_file /home/vault/iwi5/iwi5165h/DINO-Checkpoints/checkpoint_resnet50_DINO_Training_Job_036_ResNet50_0200.pth
```

### ResNet50 Test (Alpha)
```
srun python main.py --backbone medical_resnet50 --dataset_reduction 10 --alpha_correspondence --lr 0.00004 --lr_backbone 0.000004 --batch_size 2 --epochs 5 --lr_drop_epochs 2 --with_box_refine --two_stage --eff_query_init --eff_specific_head --rho 0.1 --use_enc_aux_loss --num_queries 300 --data_dir /home/vault/iwi5/iwi5165h/2024-04-Scheuplein-Screw-Detection --result_dir /home/hpc/iwi5/iwi5165h/Screw-Detection-Results --backbone_checkpoint_file /home/vault/iwi5/iwi5165h/DINO-Checkpoints/checkpoint_resnet50_DINO_Training_Job_036_ResNet50_0200.pth
```

### ViT-S-16 Test
```
```
