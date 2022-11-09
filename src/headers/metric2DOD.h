#ifndef Metric2DOD_H
#define Metric2DOD_H
#pragma once

#include <QThread>
#include <QString>
#include <QStringList>
#include <QObject>
#include "bboxes.h"

using namespace std;

class Metric2DOD : public QThread
{
    Q_OBJECT
public:
    QString now_img_data_path;
    int class_cnt;
    QStringList class_list;
    struct IoUInfo
    {
        int cls;
        float conf;
        float iou;
    };

public:
    Metric2DOD(QObject *parent = 0);
    void setPaths(string);
    vector<BBoxes::BBox2D> getLabelVector(QString);
    pair<float, float> returnAvgIOU(vector<BBoxes::BBox2D>, vector<BBoxes::BBox2D>, vector<BBoxes::BBox2D>);
    int *calcBoxes(float, float, float, float, int, int);
    void calcMetrics();

private:
    string gt_label, net1_label, net2_label;
    QStringList gt_label_list, net1_label_list, net2_label_list;

    vector<Metric2DOD::IoUInfo> *cls_iou_info1;
    vector<Metric2DOD::IoUInfo> *cls_iou_info2;
    int *cls_gt;
    int *obj_size_list;
    int volume;

private:
    vector<Metric2DOD::IoUInfo> calcIOUwithNetandGT(vector<BBoxes::BBox2D>, vector<BBoxes::BBox2D>);
    float calcIOU(BBoxes::BBox2D, BBoxes::BBox2D);
    void orgByClass(int, vector<Metric2DOD::IoUInfo>);
    vector<pair<float, float>> *accTPFP(vector<Metric2DOD::IoUInfo> *, float);
    vector<pair<QString, float>> calcAPbyCOCO(vector<pair<float, float>> *);
    vector<pair<QString, float>> calcAPbyVOC(vector<pair<float, float>> *);
    float getAverageIOU(vector<Metric2DOD::IoUInfo>);
    void getObjSizeList(vector<BBoxes::BBox2D>);
    int *getClassConfList(vector<Metric2DOD::IoUInfo>);
    bool checkIndex(vector<int>, int);
    void initializeVecArray(vector<pair<float, float>> *);
    float calcClassConfVar(vector<Metric2DOD::IoUInfo>);
    float calcObjSimilarity(vector<Metric2DOD::IoUInfo> *, vector<Metric2DOD::IoUInfo> *);
    float calcBBoxAcc(vector<float>);
    int calcBoxSizeIndicator(BBoxes::BBox2D);
    float calcNormVariance(int *, int, int);

signals:
    void sendmAPs(float, float);
    void sendNetAPs(int, vector<pair<QString, float>>);
    void sendAvgIOUs(vector<float>, vector<float>);
    void sendBBoxAcc(float, float, float);
    void sendVariance(float, float);
    void sendObjSim(float);

public:
    ~Metric2DOD();
};

#endif