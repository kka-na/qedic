#ifndef MODE2DOD_h
#define MODE2DOD_h
#pragma once

#include "metric2DOD.h"
 

#include <QThread>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QImage>

using namespace std;

class mode2DOD : public QThread
{
    Q_OBJECT
public:
    explicit mode2DOD(QObject *parent = 0);
    void setData(string, QString, QString);
    void saveAccept(string);
    void saveReject(string);
    int n1_w, n1_h, gt_w, gt_h;
    QStringList class_list;
    int class_cnt;
    int threshold;
    Metric2DOD *met2DOD;

private:
    int dir_size;

    string dataset_path;
    QString data_path, label_path;
    QStringList img_data_list, label_data_list;
    int now_data_index;
    string now_img_data_name, now_label_data_name;
    QString now_img_data_path;
    QString net1_label_path, net2_label_path, gt_label_path;

    
    int n1_obj_count, n2_obj_count, gt_obj_count = 0;

private:
    int* calcBoxes(float, float, float, float, int, int);
    void setBoxes();
    vector<BBoxes::BBox2D> getLabelVector(QString);
    void drawBoxes(int, vector<BBoxes::BBox2D>);
    QPen getBBoxPen(int);
    void setClassName();

signals:
    void sendStart();
    void sendStop();
    void sendGTImg(QImage);
    void sendNet1Img(QImage);
    void sendNet2Img(QImage);
    void sendImgList(QStringList);
    void sendAvgIOU(float, float);

private slots:
    void calcAccuracy();
    void setThreshold(int);
    void setDataIdx(int);
    void goLeft();
    void goRight();

};

#endif