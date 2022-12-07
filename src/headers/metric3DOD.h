#ifndef Metric3DOD_H
#define Metric3DOD_H
#pragma once

#include <QThread>
#include <QString>
#include <QStringList>
#include "bboxes.h"

using namespace std;

class Metric3DOD : public QThread
{
    Q_OBJECT
public:
    QString now_pcd_data_path;
    int class_cnt;
    QStringList class_list;
    struct IoUInfo
    {
        int cls;
        float conf;
        float iou;
    };

public:
    Metric3DOD(QObject *parent = 0);
    BBoxes clsBBoxes;
    void setPaths(string);
    vector<BBoxes::BBox3D> getLabelVector(int, QString);
    pair<float, float> returnAvgIOU(vector<BBoxes::BBox3D>, vector<BBoxes::BBox3D>, vector<BBoxes::BBox3D>);
    void calcMetrics();

private:
    string gt_label, net1_label, net2_label;
    QStringList gt_label_list, net1_label_list, net2_label_list;
    vector<Metric3DOD::IoUInfo> *cls_iou_info1;
    vector<Metric3DOD::IoUInfo> *cls_iou_info2;
    int *cls_gt;
    int *obj_size_list;
    int volume;

private:
    int getIdxbyCls(string);
    vector<Metric3DOD::IoUInfo> calcIOUwithNetandGT(vector<BBoxes::BBox3D>, vector<BBoxes::BBox3D>);
    float calcIOU(BBoxes::BBox3D, BBoxes::BBox3D);
    pair<float, float> calcXY(int, BBoxes::Corner3D, BBoxes::Corner3D, BBoxes::Corner3D, BBoxes::Corner3D);
    void orgByClass(int, vector<Metric3DOD::IoUInfo>);
    // void accTPFP(int, vector<Metric3DOD::IoUInfo>);
    vector<pair<float, float>> *accTPFP(vector<Metric3DOD::IoUInfo> *, float);
    vector<pair<QString, float>> calcAPbyClass(vector<pair<float, float>> *);
    float getAverageIOU(vector<Metric3DOD::IoUInfo>);
    void getObjSizeList(vector<BBoxes::BBox3D>);
    int *getClassConfList(vector<Metric3DOD::IoUInfo>);
    bool checkIndex(vector<int>, int);
    void initializeVecArray(vector<pair<float, float>> *);
    float calcClassConfVar(vector<Metric3DOD::IoUInfo>);
    float calcObjSimilarity(vector<Metric3DOD::IoUInfo> *, vector<Metric3DOD::IoUInfo> *);
    float calcBBoxAcc(vector<float>);
    int calcBoxSizeIndicator(BBoxes::BBox3D);
    float calcNormVariance(int *, int, int);

signals:
    void sendmAPs(float, float);
    void sendNetAPs(int, vector<pair<QString, float>>);
    void sendAvgIOUs(vector<float>, vector<float>);
    void sendBBoxAcc(float, float, float);
    void sendVariance(float, float);
    void sendObjSim(float);

public:
    ~Metric3DOD();
};

#endif