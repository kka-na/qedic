<p align="center">
  <img width="200" height="200" src="https://user-images.githubusercontent.com/69347961/139665124-9a8fd604-9dbb-4ff8-9b71-3f2166e55364.png"/>
</p>

# QEDiC
Quality Evaluation and Difficulty Classification of your Deep-Learning Model's Prediction Result and Ground Truth Data. 

## Environment 

* please download all libraries at home 
* or should modify CMakeLists.txt file 

### Open3D 
http://www.open3d.org/docs/release/compilation.html

Building from Source
```
git clone --recursive https://github.com/intel-isl/Open3D
git submodule update --init --recursive
cd Open3D
util/install_deps_ubuntu.sh
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=<open3d_install_directory> ..
make -j$(nproc)
make install
```

### VTK 
https://machineseez.blogspot.com/2017/06/installing-point-cloud-library-18-on.html

First, Download vtk version under 8 ( because there is no QVTKWidgetPlugin in 9 )
```
wget http://www.vtk.org/files/release/7.1/VTK-7.1.0.tar.gz
tar -xf VTK-7.1.0.tar.gz
cd VTK~
mkdir build
cd build
cmake -D VTK_QT_VERSION:STRING=5\ -D VTK_GROUP_Qt:BOOL=ON\ -D QT_QMAEK_EXECUTABLE:PATH=/usr/bin/qmake\ -D CMAKE_INSTALL_PREFIX:PATH=/usr/local .. 
**please check VTK_Group_Qt=ON
make -j$(proc)
make install
```
Second, move <QVTKWidgetPlugin> to qt 
```
cd VTK~/build/lib
sudo cp ./libQVTKWidgetPlugin.so /usr/lib/x86_64-linux-gnu/qt5/plugins/designer
```

### QWT 
https://sourceforge.net/projects/qwt/

First, Download qwt latest file
```
cd qwt~
qmake qwt.pro
make
sudo cp qwt~ /usr/local/
```

## Installation
```
git clone https://github.com/kka-na/qedic.git
cd qedic
mkdir build && cd build
cmake .. && make
./qedic
```
  
## Prepare Data
Please download the [Inha CVLab 2D Object Detection Sample Data](https://drive.google.com/file/d/1crFGflbWh7Jhk63PV5zP9DCZrnDmeByo/view?usp=sharing), [Inha CVLab 3D Object Detection Sample Data](https://drive.google.com/file/d/1m1D5FXLfNG1hv-UWH1LKVywOztAL7Dno/view?usp=sharing) and organize the downloaded files as follows. 

   ```
    sample_data_2D
    ├── gt
    │   ├── data
    │   │   │── images(.jpeg / .png) <- change pcd files if testing 3D
    │   ├── label
    │   │   │── labels(.txt)
    ├── net1
    │   ├── data
    │   │   │── images(.jpeg / .png) 
    │   ├── label
    │   │   │── labels(.txt)
    ├── net2
    │   ├── data
    │   │   │── images(.jpeg / .png) 
    │   ├── label
    │   │   │── labels(.txt)
    ├── classes.txt
  ```

## See this video for 'QEDiC' test running. 

[![Quality Evaluation & Difficulty Classification for AV Data System Testing Video](http://img.youtube.com/vi/FSMZFGWOtNg/0.jpg)](https://youtu.be/FSMZFGWOtNg) 
