#ifndef Metric3DOD_H
#define Metric3DOD_H
#pragma once

#include <QThread>
#include <QString>
#include <QStringList>
#include "bboxes.h"

using namespace std;

class Metric3DOD : public QThread{
    Q_OBJECT
public:
    QString now_pcd_data_path;
    int class_cnt;
    QStringList class_list;
    float threshold;
    struct IoUInfo{
        int cls;
        float conf;
        float iou;
    };

public:
    Metric3DOD(QObject *parent = 0);
    BBoxes clsBBoxes;
    void setPaths(string);
    vector<BBoxes::BBox3D> getLabelVector(int,QString);
    pair<float, float> returnAvgIOU(vector<BBoxes::BBox3D>, vector<BBoxes::BBox3D>, vector<BBoxes::BBox3D>);
    void calcMetrics();

private:
    string gt_label, net1_label,  net2_label;
    QStringList gt_label_list, net1_label_list, net2_label_list;
    int TP1, FP1, TP2, FP2;
    vector<pair<float, float>> pr1;
    vector<pair<float, float>> pr2;
    vector<pair<float, float>>* cls_pr1;
    vector<pair<float, float>>* cls_pr2;
    vector<Metric3DOD::IoUInfo>* cls_iou_info1;
    vector<Metric3DOD::IoUInfo>* cls_iou_info2;
    int* cls_gt;
    vector<pair<int, int>> cls_tpfp1;
    vector<pair<int, int>> cls_tpfp2;



private:
    int getIdxbyCls(string);
    vector<Metric3DOD::IoUInfo> calcIOUwithNetandGT(vector<BBoxes::BBox3D>,vector<BBoxes::BBox3D>);
    float calcIOU(BBoxes::BBox3D, BBoxes::BBox3D);
    pair<float, float> calcXY(int, BBoxes::Corner3D, BBoxes::Corner3D, BBoxes::Corner3D, BBoxes::Corner3D);
    void orgByClass(int, vector<Metric3DOD::IoUInfo>);
    void accTPFP(int, vector<Metric3DOD::IoUInfo>);
    vector<pair<QString, float>> calcAPbyClass(vector<pair<float, float>>*);
    float getAverageIOU(vector<Metric3DOD::IoUInfo>);
    bool checkIndex(vector<int>, int);
    void initializeVecArray(vector<pair<float, float>>*);

signals:
    void sendmAPs(float, float);
    void sendNetAPs(int, vector<pair<QString, float>>);
    void sendAvgIOUs(vector<float>, vector<float>);

public:
    ~Metric3DOD();

};

#endif