#include "metric2DSS.h"


#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <ctime>

#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QImage>

Metric2DSS::Metric2DSS(QObject *parent) : QThread(parent){
    
}

void Metric2DSS::setPaths(QStringList _img_data_list, QString _gt_data_path, QString _net1_data_path, QString _net2_data_path){
    img_data_list = _img_data_list;
    gt_data_path = _gt_data_path.toStdString();
    net1_data_path = _net1_data_path.toStdString();
    net2_data_path = _net2_data_path.toStdString(); 
}

void Metric2DSS::calcMetricbyFrame(QImage gt_img, QImage net1_img, QImage net2_img){
    int catergories = class_cnt * class_cnt;
    vector<int> confusion1(catergories, 0);
    vector<int> confusion2(catergories, 0);
    if(gt_img.size() != net1_img.size())
        net1_img = net1_img.scaled(gt_img.size());
    if(gt_img.size() != net2_img.size())
        net2_img = net2_img.scaled(gt_img.size());
    for(int r=0; r<gt_img.height(); r++){
        for(int c=0; c<gt_img.width(); c++){
            int gt_idx = getClassIDbyRGB((int)qRed(gt_img.pixel(c,r)), (int)qGreen(gt_img.pixel(c,r)), (int)qBlue(gt_img.pixel(c,r)));
            int net1_idx = getClassIDbyRGB((int)qRed(net1_img.pixel(c,r)), (int)qGreen(net1_img.pixel(c,r)), (int)qBlue(net1_img.pixel(c,r)));
            int net2_idx = getClassIDbyRGB((int)qRed(net2_img.pixel(c,r)), (int)qGreen(net2_img.pixel(c,r)), (int)qBlue(net2_img.pixel(c,r)));
            int net1_conf_idx = class_cnt*gt_idx + net1_idx;                
            int net2_conf_idx = class_cnt*gt_idx + net2_idx;
            confusion1[net1_conf_idx] = confusion1[net1_conf_idx] + 1;
            confusion2[net2_conf_idx] = confusion2[net2_conf_idx] + 1;
        }
    }
    vector<pair<QString, float>> net1_ious = calcIOUbyClass(confusion1); 
    vector<pair<QString, float>> net2_ious = calcIOUbyClass(confusion2);

    float net1_avg_iou = getAverageIOU(net1_ious)*100;
    float net2_avg_iou = getAverageIOU(net2_ious)*100;
    emit sendAvgIOU(net1_avg_iou, net2_avg_iou);
}

void Metric2DSS::calcMetrics(){
    vector<QImage> gts, net1s, net2s;
    for(int i=0; i<int(img_data_list.size()); i++){
        string now_img_data_name = img_data_list.at(i).toLocal8Bit().constData();
        QImage gt_img(QString::fromStdString(gt_data_path+now_img_data_name));
        QImage net1_img(QString::fromStdString(net1_data_path+now_img_data_name));
        QImage net2_img(QString::fromStdString(net2_data_path+now_img_data_name));
        gts.push_back(gt_img);
        net1s.push_back(net1_img);
        net2s.push_back(net2_img);
    }
    vector<float> net1_avg_ious; 
    vector<float> net2_avg_ious; 

    //calc confusion matrix 
    int catergories = class_cnt * class_cnt;
    vector<int> confusion1(catergories, 0);
    vector<int> confusion2(catergories, 0);
    for(size_t i=0; i<gts.size(); i++){
        QImage gt_img = gts[i];
        QImage net1_img = net1s[i];
        QImage net2_img = net2s[i];
        if(gt_img.size() != net1_img.size())
            net1_img = net1_img.scaled(gt_img.size());
        if(gt_img.size() != net2_img.size())
            net2_img = net2_img.scaled(gt_img.size());
        vector<int> frame_confusion1(catergories, 0);
        vector<int> frame_confusion2(catergories, 0);
        for(int r=0; r<gt_img.height(); r++){
            for(int c=0; c<gt_img.width(); c++){
                int gt_idx = getClassIDbyRGB((int)qRed(gt_img.pixel(c,r)), (int)qGreen(gt_img.pixel(c,r)), (int)qBlue(gt_img.pixel(c,r)));
                int net1_idx = getClassIDbyRGB((int)qRed(net1_img.pixel(c,r)), (int)qGreen(net1_img.pixel(c,r)), (int)qBlue(net1_img.pixel(c,r)));
                int net2_idx = getClassIDbyRGB((int)qRed(net2_img.pixel(c,r)), (int)qGreen(net2_img.pixel(c,r)), (int)qBlue(net2_img.pixel(c,r)));

                int net1_conf_idx = (class_cnt*gt_idx) + net1_idx;                
                int net2_conf_idx = (class_cnt*gt_idx) + net2_idx;
                confusion1[net1_conf_idx] = confusion1[net1_conf_idx] + 1;
                confusion2[net2_conf_idx] = confusion2[net2_conf_idx] + 1; 
                frame_confusion1[net1_conf_idx] = frame_confusion1[net1_conf_idx] + 1;
                frame_confusion2[net2_conf_idx] = frame_confusion2[net2_conf_idx] + 1; 
            }
        } 
        vector<pair<QString, float>> net1_ious_frame = calcIOUbyClass(frame_confusion1); 
        vector<pair<QString, float>> net2_ious_frame = calcIOUbyClass(frame_confusion2);
        net1_avg_ious.push_back(getAverageIOU(net1_ious_frame)*100);
        net2_avg_ious.push_back(getAverageIOU(net2_ious_frame)*100);
    }
    vector<pair<QString, float>> net1_ious = calcIOUbyClass(confusion1); 
    vector<pair<QString, float>> net2_ious = calcIOUbyClass(confusion2);

    int net1_cls_num = 0;
    float net1_cls_iou_sum = 0.0;
    for(size_t i=0; i<net1_ious.size(); i++){
        if(net1_ious[i].first.toStdString() != "undefined" && net1_ious[i].second > 0.0){
            net1_cls_num ++;
            net1_cls_iou_sum = net1_cls_iou_sum + net1_ious[i].second;
        }
    }

    int net2_cls_num = 0;
    float net2_cls_iou_sum = 0.0;
    for(size_t i=0; i<net2_ious.size(); i++){
        if(net2_ious[i].first.toStdString() != "undefined" && net2_ious[i].second > 0.0){
            net2_cls_num ++;
            net2_cls_iou_sum = net2_cls_iou_sum + net2_ious[i].second;
        }
    }
    emit sendmIOUs((net1_cls_iou_sum/net1_cls_num)*100, (net2_cls_iou_sum/net2_cls_num)*100);
    emit sendNetIOUs(1, net1_ious);
    emit sendNetIOUs(2, net2_ious);
    emit sendAvgIOUs(net1_avg_ious, net2_avg_ious);
}

vector<pair<QString, float>> Metric2DSS::calcIOUbyClass(vector<int> confusion){
    vector<pair<QString, float>> ious;
    
    for(int i=0; i<class_cnt; i++){
        int union_sum = 0; int intersection = 0;
        int col_sum = 0; int row_sum = 0;
        for(int j=0; j<class_cnt; j++){
            col_sum = col_sum + confusion[(i + (class_cnt * j))];
            row_sum = row_sum + confusion[((i * class_cnt) + j)];
        }   
        intersection = confusion[(i * class_cnt + i)];
        union_sum = col_sum + row_sum - intersection;
        //cout<<intersection<<" "<<union_sum<<endl;
        float iou;
        if(union_sum == 0 && intersection == 0 ) iou = 0.0;
        else iou = intersection / (union_sum + 1e-9) ;
        QString name = QString::fromStdString(semantic_classes[i].name);
        ious.push_back(make_pair(name, iou));
    }
    return ious;
}

int Metric2DSS::getClassIDbyRGB(int r, int g, int b){
    int find_idx = 11;
    for(int i=0; i<int(semantic_classes.size()); i++){
        if(semantic_classes[i].r == r && semantic_classes[i].g == g && semantic_classes[i].b == b)
            find_idx = semantic_classes[i].id;
    }
    return find_idx;
}

float Metric2DSS::getAverageIOU(vector<pair<QString, float>> ious){
    float sum = 0.0;
    int cnt = 0;
    for(size_t i=0; i<ious.size(); i++){
        if(ious[i].second > 0.0){
            cnt++;
            sum = sum + ious[i].second;
        }
    }
    return sum / (float(cnt) + 1e-9);
}

void Metric2DSS::setClassVectors(){
    for(int i=0; i<int(class_list.size()); i++){
        Metric2DSS::Semantics semantic;
        semantic.id = i;
        semantic.name = class_list[i].first;
        int *rgb = convHextoRGB(class_list[i].second);
        semantic.r = rgb[0];
        semantic.g = rgb[1];
        semantic.b = rgb[2];
        semantic_classes.push_back(semantic);
    }
}

int* Metric2DSS::convHextoRGB(string hexCol){
    static int rgb[3];
    int r,g,b;
    sscanf(hexCol.c_str(), "#%02x%02x%02x", &r, &g, &b);
    rgb[0] = r; rgb[1] = g; rgb[2] = b;
    return rgb;
}
