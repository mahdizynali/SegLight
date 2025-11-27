## Inference On C++
First install tensorflow c_lang for cpu which size is 121MB :
```
https://www.tensorflow.org/install/lang_c
```
In order to make and run this inferencer via cpp, you have to install CppFlow at first as it perform us to inference \
tensorflow models with c++ without installing it actually.
```
git clone git@github.com:serizba/cppflow.git
cd cppflow/examples/load_model
mkdir build
cd build
cmake ..
make -j
sudo make install
```
Also we need Eigen linear algebra library :
```
sudo apt install libeigen3-dev
```
Fine now try to build and inference your model, default is webcam 0, so may use other camera or single image as i declare functions.
```
mkdir build && cd build
cmake ..
make -j`nproc`
./inference
```
