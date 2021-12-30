#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "ui_mainwindow.h"

#include "vtkWidget.h"
#include "Timestamp.h" 
#include "mode2DOD.h"
#include "mode3DOD.h"
#include "mode2DSS.h"

#include <QMainWindow>
#include <QObject>
#include <QWidget>
#include <QLabel>
#include <QString>
#include <QStringList>

using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public: 
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    class mode2DOD::mode2DOD *od2d;
    class mode3DOD::mode3DOD *od3d;
    class mode2DSS::mode2DSS *ss2d;
    class vtkWidget::vtkWidget *vtkW1;
    class vtkWidget::vtkWidget *vtkW2;
    class vtkWidget::vtkWidget *vtkWg;

private:
	Ui::MainWindow *ui;
    Timestamp ts;

    string storage_path;
    string dataset_path;
    QString data_path, label_path;

    string server, task;
    int threshold;
    int task_idx; //0: 2D O.D, 1: 2D S.S, 2: 3D O.D

    QImage gtImg;
    QString gtPCD_path;
    vector<BBoxes::BBox3D> gtPCD_boxes;

    float avg_net1, avg_net2;

    float net1_achieve, net2_achieve;

private:
    void setFunction();
    void setEachTasks();
    int getLOD(float, float);
    void setLOD();
    void setAchieve(float, float);
    void set2DODLayouts();
    void set2DSSLayouts();
    void set3DODLayouts();
    void clearLayouts();

private slots:
    void setData();
    void setListIdx(QListWidgetItem*);
    void setImageList(QStringList);
    void setPCDList(QStringList);
    void setGTImage(QImage); 
    void setNet1Image(QImage);
    void setNet2Image(QImage);
    void setGTPCD(QString, vector<BBoxes::BBox3D>);
    void setAvgIOU(float, float);
    void setLoadingMovie();
    void stopLoadingMovie();
    void setmAPs(float, float);
    void setNetAPs(int, vector<pair<QString, float>>);
    void setAvgIOUs(vector<float>, vector<float>);
    void setThreshold(int);
    void setServer();
    void setTask();
    void setStorage();
    void displayDetail();
    void dataAccept();
    void dataReject();

signals:
    void sendGTPCD(QString, vector<BBoxes::BBox3D>);
    void sendListIdx(int);

private:
    QLabel* gtLabel;
    QLabel* net1Label;
    QLabel* net2Label;
    QLabel* loadingLabel;
    QMovie* movie;
    
};
#endif //MAINWINDOW_H
