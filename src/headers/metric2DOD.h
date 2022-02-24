#ifndef Metric2DOD_H
#define Metric2DOD_H
#pragma once

#include <QThread>
#include <QString>
#include <QStringList>
#include <QObject>
#include "bboxes.h"

using namespace std;

class Metric2DOD : public QThread{
    Q_OBJECT
public:
    QString now_img_data_path;
    int class_cnt;
    QStringList class_list;
    float threshold;
    struct IoUInfo{
        int cls;
        float conf;
        float iou; 
    };

public:
    Metric2DOD(QObject *parent = 0);
    void setPaths(string);
    vector<BBoxes::BBox2D> getLabelVector(QString);
    pair<float, float> returnAvgIOU(vector<BBoxes::BBox2D>, vector<BBoxes::BBox2D>, vector<BBoxes::BBox2D>);
    int* calcBoxes(float, float, float, float, int, int);
    void calcMetrics();

private:
    string gt_label, net1_label,  net2_label;
    QStringList gt_label_list, net1_label_list, net2_label_list;
    int TP1, FP1, TP2, FP2;
    vector<pair<float, float>> pr1;
    vector<pair<float, float>> pr2;
    vector<pair<float, float>>* cls_pr1;
    vector<pair<float, float>>* cls_pr2;
    vector<Metric2DOD::IoUInfo>* cls_iou_info1;
    vector<Metric2DOD::IoUInfo>* cls_iou_info2;
    int* cls_gt;
    vector<pair<int, int>> cls_tpfp1;
    vector<pair<int, int>> cls_tpfp2;

private:
    vector<Metric2DOD::IoUInfo> calcIOUwithNetandGT(vector<BBoxes::BBox2D>,vector<BBoxes::BBox2D>);
    float calcIOU(BBoxes::BBox2D, BBoxes::BBox2D);
    void orgByClass(int, vector<Metric2DOD::IoUInfo>);
    void accTPFP(int, vector<Metric2DOD::IoUInfo>);
    vector<pair<QString, float>> calcAPbyClass(vector<pair<float, float>>*);
    float getAverageIOU(vector<Metric2DOD::IoUInfo>);
    bool checkIndex(vector<int>, int);
    void initializeVecArray(vector<pair<float, float>>*);


signals:
    void sendmAPs(float, float);
    void sendNetAPs(int, vector<pair<QString, float>>);
    void sendAvgIOUs(vector<float>, vector<float>);

public:
    ~Metric2DOD();

};

#endif