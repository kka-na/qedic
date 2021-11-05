#ifndef MODE2DSS_h
#define MODE2DSS_h
#pragma once

#include "metric2DSS.h"

#include <QThread>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QImage>

using namespace std;

class mode2DSS : public QThread
{
    Q_OBJECT
public:
    explicit mode2DSS(QObject *parent = 0);
    void setData(string);
    vector<pair<string,string>> class_list;
    int class_cnt;
    int threshold;
    Metric2DSS *met2DSS;

private:
    int dir_size;
    string dataset_path;
    QString gt_data_path, net1_data_path, net2_data_path;
    QStringList img_data_list; 
    int now_data_index;
    string now_img_data_name;
    QString net1_now_img_data_path, net2_now_img_data_path, gt_now_img_data_path;

private:
    void setPolygons();
    void setClassList();

signals:
    void sendStart();
    void sendStop();
    void sendGTImg(QImage);
    void sendNet1Img(QImage);
    void sendNet2Img(QImage);
    void sendImgList(QStringList);
    //void sendAvgIOU(float, float);

private slots:
    void calcAccuracy();
    void setThreshold(int);
    void setDataIdx(int);
    void goLeft();
    void goRight();
};

#endif