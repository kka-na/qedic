#ifndef Metric2DSS_H
#define Metric2DSS_H
#pragma once 

#include <QThread>
#include <QString>
#include <QStringList>
#include <QObject>

using namespace std;

class Metric2DSS : public QThread{
    Q_OBJECT

public:
    Metric2DSS(QObject *parent = 0);
    void setPaths(QStringList, QString, QString, QString);
    void setClassVectors();
    void calcMetrics();
    void calcMetricbyFrame(QImage, QImage, QImage);

public:
    struct Semantics{
        int id;
        string name;
        int r;
        int g;
        int b;
    };
    int class_cnt; 
    vector<pair<string,string>> class_list;
    float threshold;
     
private:
    int* convHextoRGB(string);
    void initialConfVec();
    int getClassIDbyRGB(int, int, int);
    vector<pair<QString, float>> calcIOUbyClass(vector<int>);
    float getAverageIOU(vector<pair<QString, float>>);
    
private:
    string gt_data_path, net1_data_path, net2_data_path;
    QStringList img_data_list;
    vector<Metric2DSS::Semantics> semantic_classes;

signals:
    void sendmIOUs(float, float);
    void sendNetIOUs(int, vector<pair<QString, float>>);
    void sendAvgIOU(float, float);
    void sendAvgIOUs(vector<float>, vector<float>);
};

#endif