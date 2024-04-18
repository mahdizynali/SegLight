## Light Semantic Segmentation
Approch of this project is implementing light weight semantic segmentation model which has the least inference time as possible on cpu that we use on humanoid soccer robots. Also network architecture has been inspired from our last team project.
## How to use
first of all you have to prepare suitable semantic dataset as images and labels files in dataset directory like bellow :
```
datase/
  |_ /images/
  |_ /labels/
  |_ classes.json (if you want !)
```
![alt text](https://raw.githubusercontent.com/mahdizynali/SegLight/main/dataset/images/new_46.png)
![alt text](https://github.com/mahdizynali/SegLight/blob/main/dataset/labels/new_46.png) \
Then you have to set your configuration in config file and intiate your semantic color-map.
#### Notice : if you don't have reach dataset, you would use repeat option in augmentation data_provider file :
```
train_dataset = train_dataset.repeat(60)
repeat dataset 60 times !!
```
Finally try to run main.py as trainer file to store trained model into that specific folder which you set in main.py.

```
@software{Mahdi_SegLight_Light_Semantic,
  author = {Mahdi, Zeinali},
  title = {{SegLight (Light Semantic Segmentation For Humanoid Soccer Robots)}},
  url = {https://github.com/mahdizynali/SegLight},
  version = {1.0}
}
```
