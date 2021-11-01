#ifndef MODE3DOD_h
#define MODE3DOD_h
#pragma once

#include "metric3D.h"

#include <QThread>
#include <QObject>
#include <QString>
#include <QStringList>

using namespace std;

class mode3DOD : public QThread
{
    Q_OBJECT

public:
    explicit mode3DOD(QObject *parent=0);
    void setData(string, QString, QString);
    QStringList class_list;
    int class_cnt;
    int threshold;
    Metric3D *met3D;

private:
    int dir_size;
    string dataset_path;
    QString data_path, label_path;
    QStringList pcd_data_list, label_data_list;
    int now_data_index;
    string now_pcd_data_name, now_label_data_name;
    QString now_pcd_data_path;
    QString net1_label_path, net2_label_path, gt_label_path;

    int n1_obj_count, n2_obj_count, gt_obj_count = 0;

private :
    void setBoxes();
    vector<BBoxes::BBox3D> getLabelVector(QString); 
    void setClassName();

signals:
    void sendPCDList(QStringList);
    void sendAvgIOU(float, float);
    void sendGTPCD(QString, vector<BBoxes::BBox3D>);
    void sendNet1PCD(QString, vector<BBoxes::BBox3D>);
    void sendNet2PCD(QString, vector<BBoxes::BBox3D>);

private slots:
    void calcAccuracy();
    void setThreshold(int);
    void setDataIdx(int);
    void goLeft();
    void goRight();
};

#endif 