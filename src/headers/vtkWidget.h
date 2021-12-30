#ifndef VTKWIDGET_H
#define VTKWIDGET_H

#include "bboxes.h"

#include <string> 

#include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/Geometry3D.h>
#include <open3d/visualization/visualizer/O3DVisualizer.h>

#include <QWidget>
#include <QVTKWidget.h>
#include <QObject>
#include <QString>
#include <Eigen/Eigen>
#include <vtkCamera.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <boost/shared_ptr.hpp>

using namespace std; 

class vtkWidget : public QVTKWidget
{
    Q_OBJECT
public:
    vtkWidget(QWidget *parent = 0);
    BBoxes clsBBoxes;
    void init();

    boost::shared_ptr<open3d::geometry::PointCloud> defcloud;
    vtkSmartPointer<vtkCamera> camera;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor;

private:
    void dispPointCloud(string, vector<BBoxes::Corners3D>, vector<BBoxes::Corner3D>, vector<float>, vector<int>);
    vtkSmartPointer<vtkUnstructuredGrid> MakeHexa(BBoxes::Corners3D);
    double* getColors(int);

public slots:
    void display_pcd(QString, vector<BBoxes::BBox3D>);

};
#endif //VTKWIDGET_H