#include "vtkWidget.h"

#include <QThread> 
#include <QVTKInteractor.h>
#include <QMetaType>

#include <atomic>
#include <cassert>
#include <cmath>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <vector>
#include <memory>

#include <vtkDataSetMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkBox.h>
#include <vtkPolygon.h>
#include <vtkPolyData.h>
#include <vtkHexahedron.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkGlyph3DMapper.h>
#include <vtkNew.h>
#include <vtkActor.h>
#include <vtkNamedColors.h>
#include <vtkCubeSource.h>
#include <vtkPointSource.h>
#include <vtkBoundingBox.h>
#include <vtkCellArray.h>
#include <vtkTransform.h>

using namespace std;

vtkWidget::vtkWidget(QWidget* parent) : QVTKWidget(parent){
   this->SetRenderWindow(renderWindow.Get());
   this->renderWindowInteractor = this->GetInteractor();
}

void vtkWidget::init(){    
    camera = vtkSmartPointer<vtkCamera>::New();
    camera->SetViewUp(0,1,0);
    camera->SetPosition(0,0,150);
    camera->SetFocalPoint(0,0,0);

    renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetActiveCamera(camera);
    renderer->SetBackground(0.239, 0.262, 0.341);
    this->GetRenderWindow()->AddRenderer(renderer);
}

void vtkWidget::dispPointCloud(string fileName, vector<BBoxes::Corners3D> corners, vector<BBoxes::Corner3D> positions, vector<float> yaws, vector<int> classes){ 
    //Points 
    defcloud.reset( new open3d::geometry::PointCloud );
    open3d::io::ReadPointCloud(fileName, *defcloud);
    std::vector<Eigen::Vector3d> cloud_points = defcloud->points_;
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for(int i=0; i<int(cloud_points.size()); i++){
        points->InsertNextPoint(cloud_points.at(i)(0), cloud_points.at(i)(1), cloud_points.at(i)(2));
    }
    vtkSmartPointer<vtkPolyData> poly_data = vtkSmartPointer<vtkPolyData>::New();
    poly_data->SetPoints(points);
    vtkSmartPointer<vtkVertexGlyphFilter> glyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    glyphFilter->SetInputData(poly_data);
    glyphFilter->Update();
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(glyphFilter->GetOutputPort());
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetPointSize(0.01);
    actor->GetProperty()->SetColor(0.886, 0.901, 0.921);
    
    //Boxes
    vector<vtkSmartPointer<vtkUnstructuredGrid>> uGrids;
    vector<vtkSmartPointer<vtkDataSetMapper>> mappers;
    vector<vtkSmartPointer<vtkActor>> actors;

 
    for(int i=0; i<int(corners.size()); i++){
        uGrids.push_back(MakeHexa(corners[i]));
    }
    for(int i=0; i<int(uGrids.size()); i++){
        mappers.push_back(vtkSmartPointer<vtkDataSetMapper>::New());
        actors.push_back(vtkSmartPointer<vtkActor>::New());
        mappers[i]->SetInputData(uGrids[i]);
        vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
        transform->Translate(positions[i].x, positions[i].y, positions[i].z);
        transform->RotateZ(yaws[i]*180/3.141592); //kitti -pi ~ pi
        transform->Translate(-positions[i].x, -positions[i].y, -positions[i].z);
        actors[i]->SetMapper(mappers[i]);
        actors[i]->SetUserMatrix(transform->GetMatrix());
        actors[i]->GetProperty()->SetColor(this->getColors(classes[i]));
        actors[i]->GetProperty()->SetOpacity(0.5);
    }
    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderer->AddActor(actor);
    for(int i=0; i<int(uGrids.size()); i++){
        renderer->AddActor(actors[i]);
    }
    this->GetRenderWindow()->Render();   
}

vtkSmartPointer<vtkUnstructuredGrid> vtkWidget::MakeHexa(BBoxes::Corners3D corners){
  // A voxel is a representation of a regular grid in 3-D space.
  int numberOfVertices = 8;
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  
  points->InsertNextPoint(corners.cor0.x, corners.cor0.y, corners.cor0.z);
  points->InsertNextPoint(corners.cor1.x, corners.cor1.y, corners.cor1.z);
  points->InsertNextPoint(corners.cor2.x, corners.cor2.y, corners.cor2.z);
  points->InsertNextPoint(corners.cor3.x, corners.cor3.y, corners.cor3.z);
  points->InsertNextPoint(corners.cor4.x, corners.cor4.y, corners.cor4.z);
  points->InsertNextPoint(corners.cor5.x, corners.cor5.y, corners.cor5.z);
  points->InsertNextPoint(corners.cor6.x, corners.cor6.y, corners.cor6.z);
  points->InsertNextPoint(corners.cor7.x, corners.cor7.y, corners.cor7.z);

  vtkSmartPointer<vtkHexahedron> hexahed = vtkSmartPointer<vtkHexahedron>::New();
  for (int i = 0; i < numberOfVertices; ++i){
    hexahed->GetPointIds()->SetId(i, i);
  }

  vtkSmartPointer<vtkUnstructuredGrid> uGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  uGrid->SetPoints(points);
  uGrid->InsertNextCell(hexahed->GetCellType(), hexahed->GetPointIds());

  return uGrid;
}


void vtkWidget::display_pcd(QString fileName, vector<BBoxes::BBox3D> bboxes){ //Const& ::Ptr
    this->init(); 
    vector<BBoxes::Corners3D> corners;
    vector<BBoxes::Corner3D> positions;
    vector<float> yaws;
    vector<int> classes;
    for(int i=0; i<int(bboxes.size()); i++){
        corners.push_back(clsBBoxes.calcCorners(bboxes[i]));
        classes.push_back(bboxes[i].cls);
        BBoxes::Corner3D pos;
        pos.x = bboxes[i].cx; pos.y = bboxes[i].cy; pos.z = bboxes[i].cz;
        positions.push_back(pos);
        yaws.push_back(bboxes[i].yaw);
    }
    this->dispPointCloud(fileName.toStdString(), corners, positions, yaws, classes);

}

double* vtkWidget::getColors(int cls){
    static double p[3];
    if (cls % 5 == 0){
        p[0] = double(0.584);
        p[1] = double(0.501);
        p[2] = double(1.0);
    }else if (cls % 5 == 1){
        p[0] = double(0.498);
        p[1] = double(1.0);
        p[2] = double(0.917);
    }else if (cls % 5 == 2){
        p[0] = double(0.956);
        p[1] = double(0.498);
        p[2] = double(0.749);
    }else if (cls % 5 == 3){
        p[0] = double(1.0);
        p[1] = double(0.721);
        p[2] = double(0.423);
    }else if (cls % 5 == 4){
        p[0] = double(0.545);
        p[1] = double(0.913);
        p[2] = double(0.992);
    }
    return p;
}
 

    

