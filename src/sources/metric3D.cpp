#include "metric3D.h"

#include <QFile>
#include <QDir>
#include <QMetaType>
#include <QTextStream>

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>	

#include <jsoncpp/json/json.h>


Metric3D::Metric3D(QObject *parent) : QThread(parent){
    
}

void Metric3D::setPaths(string _dataset_path){
    gt_label = _dataset_path+"/gt/label/"; net1_label = _dataset_path+"/net1/label/"; net2_label = _dataset_path+"/net2/label/";
    QString gt_label_path = QString::fromStdString(gt_label);
    QString net1_label_path = QString::fromStdString(net1_label);
    QString net2_label_path = QString::fromStdString(net2_label);

    QDir gt_label_dir(gt_label_path);
    QDir net1_label_dir(net1_label_path);
    QDir net2_label_dir(net2_label_path);

    gt_label_list = gt_label_dir.entryList(QStringList() << "*.json");
    net1_label_list = net1_label_dir.entryList(QStringList()<<"*.json");
    net2_label_list = net2_label_dir.entryList(QStringList()<<"*.json");
}

pair<float, float> Metric3D::returnAvgIOU(vector<BBoxes::BBox3D> _gtVec, vector<BBoxes::BBox3D> _net1Vec, vector<BBoxes::BBox3D> _net2Vec){
    vector<pair<int, float>> net1_ious = this->calcIOUwithNetandGT(_gtVec, _net1Vec);  
    vector<pair<int, float>> net2_ious = this->calcIOUwithNetandGT(_gtVec, _net2Vec);  
    float net1_avg_iou = this->getAverageIOU(net1_ious);
    float net2_avg_iou = this->getAverageIOU(net2_ious);
    return make_pair(net1_avg_iou, net2_avg_iou);
}

vector<BBoxes::BBox3D> Metric3D::getLabelVector(QString label_path){
    vector<BBoxes::BBox3D> vec;
    Json::Value pred;
    ifstream in(label_path.toStdString().c_str());
    if(in.is_open()) in >> pred;
    for(int i=0; i<int(pred.size()); i++){
        BBoxes::BBox3D bboxes;
        bboxes.cx = pred[i]["psr"]["position"]["x"].asFloat();
        bboxes.cy = pred[i]["psr"]["position"]["y"].asFloat();
        bboxes.cz = pred[i]["psr"]["position"]["z"].asFloat();
        bboxes.dx = pred[i]["psr"]["scale"]["x"].asFloat();
        bboxes.dy = pred[i]["psr"]["scale"]["y"].asFloat();
        bboxes.dz = pred[i]["psr"]["scale"]["z"].asFloat();
        if(pred[i]["obj_id"].size() == 0) bboxes.cls = this->getIdxbyCls(pred[i]["obj_type"].asString());
        else bboxes.cls = pred[i]["obj_id"].asInt();
        vec.push_back(bboxes);
    }
    return vec;
}

int Metric3D::getIdxbyCls(string cls){
    int idx;
    for(int i=0; i<class_cnt; i++){
        if( class_list[i].toStdString() == cls ) {
            idx = i;
            break;
        }
        else idx = 0;
    }
    return idx;
}

vector<pair<int, float>> Metric3D::calcIOUwithNetandGT(vector<BBoxes::BBox3D> gtVec, vector<BBoxes::BBox3D> netVec){
    vector<int> checked;
    vector<pair<int, float>> ious;
    for(size_t i=0; i<netVec.size(); i++){
        int temp_min = 99999999; 
        int gtidx, netidx = 0;
        for(size_t j=0; j<gtVec.size(); j++){
            int min = abs(gtVec[j].cx-netVec[i].cx)+abs(gtVec[j].cy-netVec[i].cy)+abs(gtVec[j].cz-netVec[i].cz);
            if(temp_min>min){
                temp_min = min;
                gtidx = j; netidx = i;
            }else{
                continue;
            }
        }
        float iou = this->getIOU(gtVec[gtidx], netVec[netidx]);
        ious.push_back(make_pair(gtVec[gtidx].cls, iou));
    }
    return ious;
}

float Metric3D::getIOU(BBoxes::BBox3D gt, BBoxes::BBox3D net){
    BBoxes::Corners3D gt_corner = clsBBoxes.calcCorners(gt);
    BBoxes::Corners3D net_corner = clsBBoxes.calcCorners(net);

    pair<float, float> x_plane = this->calcXY(gt_corner.cor5, gt_corner.cor0, net_corner.cor5, net_corner.cor0);
    pair<float, float> y_plane = this->calcXY(gt_corner.cor4, gt_corner.cor3, net_corner.cor4, net_corner.cor3);
    pair<float, float> z_plane = this->calcXY(gt_corner.cor6, gt_corner.cor4, net_corner.cor6, net_corner.cor4);

    float iou = 0;
    if( x_plane.second || y_plane.first || z_plane.first){
        iou = 0.0;
        return iou;
    }

    float volume_overlap = x_plane.second * y_plane.first * z_plane.first;
    float volume_gt = gt.dx * gt.dy * gt.dz ;
    float volume_net = net.dx * net.dy * net.dz;
    float volume_combined = volume_gt + volume_net - volume_overlap;

    iou = float(volume_overlap / ( volume_combined + 1e-5));    

    return iou;
}

pair<float, float> Metric3D::calcXY(BBoxes::Corner3D gtl, BBoxes::Corner3D gtr, BBoxes::Corner3D netl, BBoxes::Corner3D netr){
    float x1 = max(gtl.x, netl.x);
    float y1 = max(gtl.y, netl.y);
    float x2 = max(gtr.x, netr.x);
    float y2 = max(gtr.y, netr.y);
    float width = (x2-x1);
    float height = (y2-y1);

    return make_pair(width, height);
}


void Metric3D::accTPFP(int type, vector<pair<int, float>> ious, float gt_size){
    float precision, recall = 0.0;
    for(size_t i = 0; i<ious.size(); i++){
        if(ious[i].second <= threshold){
            if(type == 1)FP1++;
            else if(type==2)FP2++;
        }else{
            if(type == 1)TP1++;
            else if(type==2)TP2++;
        }
        if(type==1){
            precision = float(TP1) / float(TP1+FP1) + 1e-9 ;
            recall = float(TP1) / gt_size+1e-9;
            pr1.push_back(make_pair(precision, recall));
            cls_pr1[ious[i].first].push_back(make_pair(precision, recall));
        }else if(type==2){
            precision = float(TP2) / float(TP2+FP2)+ 1e-9;
            recall = float(TP2) / gt_size+1e-9; 
            pr2.push_back(make_pair(precision, recall));
            cls_pr2[ious[i].first].push_back(make_pair(precision, recall));
        }
    }
}

float Metric3D::calcAP(int type){
    float total_pr = 0.0;
    float recall_index = 10.0;
    float max_pr = 0.0;

    vector<pair<float, float>> prs;
    if(type == 1) prs = pr1;
    else if(type == 2) prs = pr2;

    while(true){
        if(recall_index < 0.0) break;
        for(size_t j=0; j<prs.size(); j++){ 
            if((recall_index/10.0)-0.1<prs[j].second && prs[j].second <= recall_index/10.0){
                if(max_pr <= prs[j].first)
                    max_pr = prs[j].first;
            }
        }
        total_pr = total_pr+max_pr;
        recall_index = recall_index - 1;
    }    
    float AP = total_pr / (11 + 1e-9) ;
    return AP;
}

vector<pair<QString, float>> Metric3D::calcAPbyClass(vector<pair<float,float>> * cls_prs){
    vector<pair<QString, float>> cls_ap(class_cnt, make_pair("-", 0.0));
    for(int i=0; i<class_cnt; i++){
        float total_pr = 0.0;
        float recall_index = 10.0;
        float max_pr = 0.0;
        while(true){
            if(recall_index<0.0) break;
            for(size_t j=0; j<cls_prs[i].size(); j++){
                if((recall_index/10.0)-0.1<cls_prs[i][j].second && cls_prs[i][j].second <= recall_index/10.0){
                    if(max_pr <= cls_prs[i][j].first)
                        max_pr = cls_prs[i][j].first;
                }
            }
            total_pr = total_pr+max_pr;
            recall_index = recall_index - 1;
        }
        float AP = total_pr / (11+1e-9);
        cls_ap.push_back(make_pair(class_list[i],AP));
    }
    return cls_ap;
}

void Metric3D::calcMetrics(){
    float all_gt_size = 0.0;
    TP1 = 0; FP1 = 0; TP2 = 0; FP2 = 0;
    
    cls_gt = new int[class_cnt]{0};
    cls_pr1 = new vector<pair<float, float>>[class_cnt]; initializeVecArray(cls_pr1);
    cls_pr2 = new vector<pair<float, float>>[class_cnt]; initializeVecArray(cls_pr2);
    vector<float> net1_avg_ious;
    vector<float> net2_avg_ious;

    for(int i=0; i<gt_label_list.size(); i++){
        vector<BBoxes::BBox3D> vecGT = getLabelVector(QString::fromStdString(gt_label+(gt_label_list.at(i).toLocal8Bit().constData())));
        all_gt_size = all_gt_size+float(vecGT.size());
        for(size_t j=0; j<vecGT.size(); j++){
            cls_gt[vecGT[j].cls] += 1;
        }
    }
    for(int i=0; i<gt_label_list.size(); i++){
        vector<BBoxes::BBox3D> vecGT = getLabelVector(QString::fromStdString(gt_label+(gt_label_list.at(i).toLocal8Bit().constData())));
        vector<BBoxes::BBox3D> vecNet1 = getLabelVector(QString::fromStdString(net1_label+(net1_label_list.at(i).toLocal8Bit().constData())));
        vector<BBoxes::BBox3D> vecNet2 = getLabelVector(QString::fromStdString(net2_label+(net2_label_list.at(i).toLocal8Bit().constData())));

        vector<pair<int, float>> net1_ious = this->calcIOUwithNetandGT(vecGT, vecNet1);  
        vector<pair<int, float>> net2_ious = this->calcIOUwithNetandGT(vecGT, vecNet2);
        net1_avg_ious.push_back((getAverageIOU(net1_ious))*100);
        net2_avg_ious.push_back((getAverageIOU(net2_ious))*100); 

        this->accTPFP(1, net1_ious, all_gt_size);
        this->accTPFP(2, net2_ious, all_gt_size);
    }
    vector<pair<QString, float>>cls_AP1 = this->calcAPbyClass(cls_pr1);
    vector<pair<QString, float>>cls_AP2 = this->calcAPbyClass(cls_pr2);

    int net1_cls_num = 0;
    float net1_cls_ap_sum = 0.0;
    for(size_t i=0; i<cls_AP1.size(); i++){
        if(cls_AP1[i].second > 0.0){
            net1_cls_num++;
            net1_cls_ap_sum = net1_cls_ap_sum+cls_AP1[i].second;
        }
    }
    int net2_cls_num = 0;
    float net2_cls_ap_sum = 0.0;
    for(size_t i=0; i<cls_AP2.size(); i++){
        if(cls_AP2[i].second>0.0){
            net2_cls_num++;
            net2_cls_ap_sum = net2_cls_ap_sum + cls_AP2[i].second;
        }
    }
    emit sendmAPs((net1_cls_ap_sum/net1_cls_num)*100, (net2_cls_ap_sum/net2_cls_num)*100);

    emit sendNetAPs(1, cls_AP1);
    emit sendNetAPs(2, cls_AP2);

    emit sendAvgIOUs(net1_avg_ious, net2_avg_ious);
}


float Metric3D::getAverageIOU(vector<pair<int, float>> vec){
    float sum = 0.0;
    for(size_t i=0; i<vec.size(); i++){
        sum = sum + vec[i].second;
    }
    return sum / float(vec.size()) + 1e-9;
}

bool Metric3D::checkIndex(vector<int> checked, int target){
    bool fine = true;
    for(size_t i = 0; i<checked.size(); i++){
        if(target == checked[i]){
            fine = false;
            break;
        }else{
            fine = true;
        }
    }
    return fine;
}

void Metric3D::initializeVecArray(vector<pair<float, float>>* vec_array){
    for(size_t i=0; i<vec_array->size(); i++){
        vec_array[i].push_back(make_pair(0.0, 0.0));
    }
}

Metric3D::~Metric3D(){
    delete [] cls_gt;
    delete [] cls_pr1;
    delete [] cls_pr2;
}

