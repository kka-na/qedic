<p align="center">
  <img width="200" height="200" src="https://user-images.githubusercontent.com/69347961/139665124-9a8fd604-9dbb-4ff8-9b71-3f2166e55364.png"/>
</p>

# QEDiC

Quality Evaluation and Difficulty Classification of your Deep-Learning Model's Prediction Result and Ground Truth Data.

---

## Newly Updated !

### v2.0

Update the Quality Assurance Indicator (QAI) calculation module.
QAI is calculated with Accuracy Achievement, Boundin-Box Accuracy, Object Similarity, Class Density and Object Size Density

### Training Result

| Dataset     | Task | Validation Size |   mAP1   |   mAP2   |
| :---------- | :--: | :-------------: | :------: | :------: |
| **COCO**    | 2DOD |      5000       | 53.6156% | 63.7698% |
| **KITTI**   | 2DOD |      1497       | 46.5362% | 51.5076% |
| **TS 2021** | 2DOD |      2891       | 51.0785% | 50.2422% |
| **KITTI**   | 3DOD |      1497       | 48.5059% | 31.3924% |
| **NUSCENs** | 3DOD |       897       | 20.4881% | 21.3303% |
| **TS 2021** | 3DOD |       500       | 41.6157% | 48.5326% |

### QAI Result

| Dataset      | Task | Accuracy Achievement | Bounding-Box Accuracy | Object Similarity | Class Density | Object Size Density |     QAI     |
| :----------- | :--: | :------------------: | :-------------------: | :---------------: | :-----------: | :-----------------: | :---------: |
| **COCO**     | 2DOD |       98.3024        |        73.0722        |      18.1468      |    3.41686    |       34.2982       | **83.1025** |
| **KITTI**    | 2DOD |         100          |        66.9405        |      23.0489      |   19.06011    |       39.6846       | **77.0294** |
| **TS 2021**  | 2DOD |       96.0166        |        66.3139        |      14.8032      |    14.3204    |       39.8156       | **78.6783** |
| **KITTI**    | 3DOD |       49.0046        |        23.3024        |      5.9789       |    30.5679    |       29.3885       | **61.2743** |
| **NUSCENES** | 3DOD |       44.0348        |        19.0174        |      27.1212      |    10.652     |       24.1427       | **60.2273** |
| **TS 2021**  | 3DOD |        55.311        |        24.8896        |      15.5961      |    43.7833    |       36.608        | **56.8426** |

### Difficulty Classification Result

| Dataset      | Task | Validation Size | IoU Threshold | Easy | Moderate | Hard |
| :----------- | :--: | :-------------: | :-----------: | :--: | :------: | :--: |
| **COCO**     | 2DOD |      5000       |      60       | 3167 |   1230   | 603  |
| **KITTI**    | 2DOD |      1497       |      60       | 814  |   402    | 281  |
| **TS 2021**  | 2DOD |      2891       |      60       | 1283 |   1424   | 184  |
| **KITTI**    | 3DOD |      1497       |      30       | 265  |   443    | 789  |
| **NUSCENES** | 3DOD |       897       |      30       | 145  |   126    | 625  |
| **TS 2021**  | 3DOD |       500       |      30       | 115  |   113    | 272  |

---

## Environment Setting

- please download all libraries at home
- or should modify CMakeLists.txt file

### Open3D

http://www.open3d.org/docs/release/compilation.html

[at implementation] version 0.11.0

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

---

## Installation

```
git clone https://github.com/kka-na/qedic.git
cd qedic
mkdir build && cd build
cmake .. && make
./qedic
```

---

## Prepare Data

Please download the
[Inha CVLab 2D Object Detection Sample Data](https://drive.google.com/file/d/1ZjGe4H0CAM18hRnk-JdDsuESKcryH4qB/view?usp=sharing),
[Inha CVLab 2D Semantic Segmentation Sample Data](https://drive.google.com/file/d/13J5iwSPK8i6tRvTEdffpQ-mb19ZNpZ5D/view?usp=sharing),
[Inha CVLab 3D Object Detection Sample Data](https://drive.google.com/file/d/13M64Sy8OkjuBaKdljpvCwo-Bx4cqENZ4/view?usp=sharing) and organize the downloaded files as follows.

2D Object Detection Test Data

```
  2DOD
  ├── gt
  │   ├── data
  │   │   │── images(.jpg / .png)
  │   ├── label
  │   │   │── labels(.txt)
  ├── net1
  │   ├── label
  │   │   │── labels(.txt)
  ├── net2
  │   ├── label
  │   │   │── labels(.txt)
  ├── classes.txt
  ├── verified
```

2D Semantic Segmentation Test Data

```
  2DSS
  ├── gt
  │   ├── data
  │   │   │── images(.jpg / .png) <- panaoptic images
  ├── net1
  │   ├── data
  │   │   │── images(.jpg / .png)
  ├── net2
  │   ├── data
  │   │   │── images(.jpg / .png)
  ├── classes.json
  ├── verified
```

3D Object Detection Test Data

```
  3DOD
  ├── gt
  │   ├── data
  │   │   │── point clouds(.pcd)
  │   ├── label
  │   │   │── labels(.json) <- sustecpoints format
  ├── net1
  │   ├── label
  │   │   │── labels(.json)
  ├── net2
  │   ├── label
  │   │   │── labels(.json)
  ├── classes.txt
  ├── verified
```

---

## See this video for 'QEDiC' test running.

[![Quality Evaluation & Difficulty Classification for AV Data System Testing Video](http://img.youtube.com/vi/duN7ffTMTec/0.jpg)](https://youtu.be/duN7ffTMTec)
