## Light Semantic Segmentation
Approch of this project is implementing light weight semantic segmentation model which has the least inference time as possible on cpu that we use on humanoid soccer robots. Also network architecture has been inspired from our last team project.
## How to use
first of all you have to prepare suitable semantic dataset as images and labels files in dataset directory like bellow :
```
datase/
  |_ seri1:
    |_ /images/
    |_ /labels/
  |_ seri2:
    |_ /images/
    |_ /labels/
  ...
```
![alt text](https://raw.githubusercontent.com/mahdizynali/SegLight/main/dataset/images/new_46.png)
![alt text](https://github.com/mahdizynali/SegLight/blob/main/dataset/labels/new_46.png) \
Then you have to set your configuration in config file and intiate your semantic color-map.
#### Notice : if you don't have reach dataset, you would use repeat option in augmentation data_provider file :
```
train_dataset = train_dataset.repeat(60)
repeat dataset 60 times !!
```
# Save Model Hint !
in tensorflow version 2.16.0 and above, keras kernel updates into version 3 and it limit us to save models only in .h5 or .keras format;
so as i wanna inference on cpp and cppflow, i need to save as tf format .pb as keras v2 in order to load in cppflow inferencer.\
try to install also keras v2 :
```
pip install tf-keras~=2.16
```
then in directory of your project set env :
```
export TF_USE_LEGACY_KERAS=1
```
in main code you would change formats as you wish :
```
model.save("save/path", save_format='tf') # for keras v2
model.save("model.h5") # or may .keras for keras v3
``` 
Finally try to run main.py as trainer file to store trained model into that specific folder which you set in main.py.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mahdizynali/SegLight&type=Date)](https://www.star-history.com/#mahdizynali/SegLight&Date)

---

## Paper Evaluation Framework

**NEW**: SegLight now includes a comprehensive evaluation framework designed for academic research and publication!

### 🔬 Research Features
- **Comprehensive Evaluation**: Segmentation metrics, performance benchmarking, statistical analysis
- **Model Comparison**: Multi-model statistical comparison with significance testing
- **Publication Ready**: Automated generation of plots, tables, and reports for academic papers
- **Statistical Rigor**: Built-in statistical tests, effect size calculation, confidence intervals

### Quick Start with Paper Framework
```bash
# Run framework demo
python simple_demo.py

# Evaluate models for paper
python paper/main_paper.py --models MazeNet --output results/

# See example usage
python example_usage.py
```

### Framework Structure
```
paper/
├── utils/           # Model management and loading
├── evaluation/      # Comprehensive metrics calculation  
├── benchmarks/      # Performance testing and profiling
├── visualization/   # Publication-ready plots and figures
├── statistics/      # Statistical analysis and testing
└── main_paper.py    # Complete evaluation pipeline
```

See `paper/README.md` for detailed documentation.

## Citation
```
@software{Mahdi_SegLight_Light_Semantic,
  author = {Mahdi, Zeinali},{Erfan, Ramezani},
  title = {{SegLight (Light Semantic Segmentation For Humanoid Soccer Robots)}},
  url = {https://github.com/mahdizynali/SegLight},
  version = {1.0}
}
```
