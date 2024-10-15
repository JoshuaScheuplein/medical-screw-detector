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

***

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

## Acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.
