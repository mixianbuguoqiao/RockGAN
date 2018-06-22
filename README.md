## RockGAN
This is a research project. It aims to generate 3D rock model using 3D Generative Adversarial Networks. 
We could control the size of rock modle. 

The environment are listed:

- python3 
- tensorflow-gpu==1.3


If you want to observe the specific effect:

```
mkdir data
mkdir ServerModel
```
you can download the trained weights in [weights](https://pan.baidu.com/s/1w9BVIahHvjsfzAD9WHYFyQ) 

put the weights in file ServerModel 

Then

```
python FlaskServer.py
```

you have to get the tool [render](https://pan.baidu.com/s/1m1j12Xh8FYpxSBrWfyTVUA) to get the render result. 

At last, run the file VoxelVisualizer2.exe.

input 127.0.0.1:5000.


If you are interested with this project. please email us. The paper is in here [3D Rock GAN](https://www.researchgate.net/publication/324111649_Masked_3D_conditional_generative_adversarial_network_for_rock_mesh_generation)
.

## Reference
https://github.com/meetshah1995/tf-3dgan
