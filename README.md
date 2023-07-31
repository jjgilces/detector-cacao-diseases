# CNN with PyTorch - Detector of deaseases in cacao
This project show how to implement a Convolutional Neural Network (CNN)using Pytorch **to detect deseases in cocoa crops** from **images**. 
The main idea is create a Neural Network capable to clasify an image in one of the following classes: black_por_rot, monilia or healthy.

**So, here’s a simple pytorch template that help you get into your main project faster and just focus on your core (Model Architecture, Training Flow, etc)**



# Requirements
- [Numpy](https://numpy.org/) (The fundamental package for scientific computing with Python
)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [Matplotlib](https://matplotlib.org/) (library for creating static, animated, and interactive visualizations)

# Table Of Contents
- [CNN with PyTorch - Detector of deaseases in cacao](#cnn-with-pytorch---detector-of-deaseases-in-cacao)
- [Requirements](#requirements)
- [Table Of Contents](#table-of-contents)
- [In a Nutshell](#in-a-nutshell)
- [In Details](#in-details)
- [Authors](#authors)

# In a Nutshell   
In a nutshell here's how to use this template, so **for example** assume you want to implement ResNet-18 to train mnist, so you should do the following:
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**


# In Details
```
├──  config
│    └── helper.py  - Utility for testing, visualizing in the contet of deep learning.
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── healthhy  - here's the datasets folder that is responsible for all data handling.
│    └── black_pod_rol  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── pod_borer   - here's the file to make dataloader.
│    └── othedeseas   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  dataset
│   ├── training     - this file contains the train loops.
│   └── test   - this file contains the inference process.
    └── validation   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```




# Authors
<div style="display: flex; ">
<a title="Diego Zambrano" href="https://github.com/dizamfi">
<img src="https://avatars.githubusercontent.com/u/69157845?v=4" width="60" style="border-radius: 50%"/>
</a>



<a title="Johan Gilces Reyes" href="https://github.com/jjgilces">
<img src="https://avatars3.githubusercontent.com/u/59465061?s=400&u=90d64167df934f58e7e1e7f5ccaba9fa6d2581cb&v=44" alt="" width="60" style="border-radius: 50%"/>
</a>

<a title="Erwin Medina" href="https://github.com/Erwing23">
<img src="https://avatars.githubusercontent.com/u/50648460?v=4" alt="Danny Loor" width="60" style="border-radius: 50%"/>
</a>

<a title="Axcel Espinoza" href="https://github.com/eapb99">
<img src="https://avatars1.githubusercontent.com/u/62962507?s=400&v=4" alt="Danny Loor" width="60" style="border-radius: 50%"/>
</a>
</div>





