#include "metric2DOD.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>

#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QImage>

Metric2DOD::Metric2DOD(QObject *parent) : QThread(parent){
    
}

void Metric2DOD::setPaths(string _dataset_path){
    gt_label = _dataset_path+"/gt/label/"; net1_label = _dataset_path+"/net1/label/"; net2_label = _dataset_path+"/net2/label/";
    QString gt_label_path = QString::fromStdString(gt_label);
    QString net1_label_path = QString::fromStdString(net1_label);
    QString net2_label_path = QString::fromStdString(net2_label);

    QDir gt_label_dir(gt_label_path);
    QDir net1_label_dir(net1_label_path);
    QDir net2_label_dir(net2_label_path);

    gt_label_list = gt_label_dir.entryList(QStringList() << "*.txt");
    net1_label_list = net1_label_dir.entryList(QStringList()<<"*.txt");
    net2_label_list = net2_label_dir.entryList(QStringList()<<"*.txt");
}

pair<float, float> Metric2DOD::returnAvgIOU(vector<BBoxes::BBox2D> _gtVec, vector<BBoxes::BBox2D> _net1Vec, vector<BBoxes::BBox2D> _net2Vec){
    vector<pair<int, float>> net1_ious = this->calcIOUwithNetandGT(_gtVec, _net1Vec);  
    vector<pair<int, float>> net2_ious = this->calcIOUwithNetandGT(_gtVec, _net2Vec);  
    float net1_avg_iou = this->getAverageIOU(net1_ious);
    float net2_avg_iou = this->getAverageIOU(net2_ious);
    return make_pair(net1_avg_iou, net2_avg_iou);
}

vector<BBoxes::BBox2D> Metric2DOD::getLabelVector(QString label_path){
    vector<BBoxes::BBox2D> vec;
	QFile label_file(label_path);
    if(!label_file.open(QIODevice::ReadOnly)){
        cout<<"The ["+label_path.toStdString()+"] was not detected."<<endl;
        BBoxes::BBox2D init = {-1,0.0,0,0,0,0,0,0,0,0};
        vec.push_back(init);
    }else{
        QTextStream in(&label_file);

        QImage target_img(now_img_data_path);
        int img_w = target_img.width();
        int img_h = target_img.height();
        while(!in.atEnd()) {
            BBoxes::BBox2D bboxes;
            QString line = in.readLine();    
            QStringList fields = line.split(" "); 
            
            bboxes.cls = fields.at(0).toInt();
            float cx, cy, w, h;
            if(fields.size() == 5){ //no confidence, this is GT
                bboxes.conf = 1.0;
                cx = fields.at(1).toFloat();
                cy = fields.at(2).toFloat();
                w = fields.at(3).toFloat();
                h = fields.at(4).toFloat();
            }else{
                bboxes.conf = fields.at(1).toFloat(); 
                cx = fields.at(2).toFloat();
                cy = fields.at(3).toFloat();
                w = fields.at(4).toFloat();
                h = fields.at(5).toFloat();
            }
            int *box = calcBoxes(cx,cy,w,h,img_w,img_h);
            
            bboxes.w = box[0]; bboxes.h = box[1];
            bboxes.lx = box[2]; bboxes.ly = box[3]; bboxes.rx = box[4]; bboxes.ry = box[5]; 
            bboxes.cx = int(cx*img_w); bboxes.cy = int(cy*img_h);
            vec.push_back(bboxes);
        }
    }
    label_file.close();
    return vec;
}

bool compare_conf(BBoxes::BBox2D a, BBoxes::BBox2D b){ return a.conf > b.conf;}

bool compare_iou(pair<int, float> a, pair<int, float> b){ return a.second > b.second;}

vector<pair<int, float>> Metric2DOD::calcIOUwithNetandGT(vector<BBoxes::BBox2D> gtVec, vector<BBoxes::BBox2D> netVec){
    vector<pair<int, float>> ious;

    sort(netVec.begin(), netVec.end(), compare_conf);
    vector<int> checked_gts = {0};
    bool detected = false;
    for(size_t i=0; i<netVec.size(); i++){
        if(netVec[i].cls == -1) continue;
        detected = true;
        vector<pair<int, float>> each_ious;
        for(size_t j=0; j<gtVec.size(); j++){
            each_ious.push_back(make_pair((int)j, this->calcIOU(gtVec[j], netVec[i])));
        }
        sort(each_ious.begin(), each_ious.end(),compare_iou);
        float iou=0.0;
        if(i == 0){
            iou = each_ious[0].second;
            checked_gts.push_back(each_ious[0].first);
        }else{
            for(size_t k=0; k<checked_gts.size(); k++){
                if(each_ious[0].first == checked_gts[k]){
                    iou = 0.0;
                }else{
                    iou = each_ious[0].second;
                    checked_gts.push_back(each_ious[0].first);
                    break;
                }    
            }
        }
        ious.push_back(make_pair(netVec[i].cls,iou));
    }
    if(!detected) ious.push_back(make_pair(-1, 0.0));
    return ious;
}

float Metric2DOD::calcIOU(BBoxes::BBox2D gt, BBoxes::BBox2D net){
    int x1 = max(gt.lx, net.lx);
    int y1 = max(gt.ly, net.ly);
    int x2 = min(gt.rx, net.rx);
    int y2 = min(gt.ry, net.ry);
    int tmpwidth = (x2-x1);
    int tmpheight = (y2-y1);

    float iou = 0;
    if(tmpwidth < 0 || tmpheight < 0){
        iou = 0.0;
        return iou;
    }
    int area_overlap = tmpwidth * tmpheight;
    int area_a = (gt.rx-gt.lx) * (gt.ry-gt.ly);
    int area_b = (net.rx-net.lx) * (net.ry-net.ly);
    int area_combined = area_a + area_b - area_overlap;
    
    iou = float(area_overlap / ( area_combined + 1e-5));    

    return iou;
}

void Metric2DOD::accTPFP(int type, vector<pair<int, float>> ious){
    for(size_t i = 0; i<ious.size(); i++){
        float precision, recall = 0.0;
        if(ious[i].first == -1) continue;
        if(ious[i].second <= threshold){
            if(type == 1) cls_tpfp1[ious[i].first].second++; //FP
            else if(type==2) cls_tpfp2[ious[i].first].second++; //FP
        }else{
            if(type == 1) cls_tpfp1[ious[i].first].first++; //TP
            else if(type==2) cls_tpfp2[ious[i].first].first++; //TP
        }
        if(type==1){
            if(cls_tpfp1[ious[i].first].first == 0 ){
                precision = 0.0;
                recall = 0.0;
            }else{
                precision = cls_tpfp1[ious[i].first].first / float(cls_tpfp1[ious[i].first].first+cls_tpfp1[ious[i].first].second);
                recall = cls_tpfp1[ious[i].first].first / float(cls_gt[ious[i].first]);
            }
            pr1.push_back(make_pair(precision, recall));
            cls_pr1[ious[i].first].push_back(make_pair(precision, recall));
        }else if(type==2){
            if(cls_tpfp2[ious[i].first].first == 0){
                precision = 0.0;
                recall = 0.0;
            }else{
                precision = cls_tpfp2[ious[i].first].first / float(cls_tpfp2[ious[i].first].first+cls_tpfp2[ious[i].first].second);
                recall = cls_tpfp2[ious[i].first].first / float(cls_gt[ious[i].first]);
            }             
            pr2.push_back(make_pair(precision, recall));
            cls_pr2[ious[i].first].push_back(make_pair(precision, recall));
        }
    }
}

vector<pair<QString, float>> Metric2DOD::calcAPbyClass(vector<pair<float,float>> * cls_prs){
    vector<pair<QString, float>> cls_ap(class_cnt, make_pair("-", 0.0));
    for(int i=0; i<class_cnt; i++){
        float total_pr = 0.0;
        float recall_index = 9.5;
        float max_pr = 0.0;
        while(true){
            if(recall_index < 5.0) break;
            for(size_t j=0; j<cls_prs[i].size(); j++){
                if((recall_index/10.0)-0.05<cls_prs[i][j].second && cls_prs[i][j].second <= recall_index/9.5){
                   //cout<<(recall_index/10.0)-0.05<<"<"<<cls_prs[i][j].second<<"&&"<<cls_prs[i][j].second<<"<="<<recall_index/9.5<<endl;
                    if(max_pr <= cls_prs[i][j].first)
                        max_pr = cls_prs[i][j].first;
                }
            }
            total_pr += max_pr;
            recall_index = recall_index - 0.5;
        }
        float AP = 0.0;
        if(total_pr == 0) AP = 0.0;
        else AP = total_pr / (10+1e-9);
        cls_ap.push_back(make_pair(class_list[i],AP));
    }
    return cls_ap;
}

void Metric2DOD::calcMetrics(){
    float all_gt_size = 0.0;
    TP1 = 0; FP1 = 0; TP2 = 0; FP2 = 0;
    
    cls_gt = new int[class_cnt]{0};
    cls_pr1 = new vector<pair<float, float>>[class_cnt]; initializeVecArray(cls_pr1);
    cls_pr2 = new vector<pair<float, float>>[class_cnt]; initializeVecArray(cls_pr2);
    cls_tpfp1 = vector<pair<int, int>>(class_cnt, make_pair(0,0));
    cls_tpfp2 = vector<pair<int, int>>(class_cnt, make_pair(0,0)); 
    vector<float> net1_avg_ious;
    vector<float> net2_avg_ious;

    for(int i=0; i<gt_label_list.size(); i++){
        vector<BBoxes::BBox2D> vecGT = getLabelVector(QString::fromStdString(gt_label+(gt_label_list.at(i).toLocal8Bit().constData())));
        all_gt_size = all_gt_size+float(vecGT.size());
        for(size_t j=0; j<vecGT.size(); j++){
            cls_gt[vecGT[j].cls] += 1;
        }
    }
    for(int i=0; i<gt_label_list.size(); i++){
        vector<BBoxes::BBox2D> vecGT = getLabelVector(QString::fromStdString(gt_label+(gt_label_list.at(i).toLocal8Bit().constData())));
        vector<BBoxes::BBox2D> vecNet1 = getLabelVector(QString::fromStdString(net1_label+(gt_label_list.at(i).toLocal8Bit().constData())));
        vector<BBoxes::BBox2D> vecNet2 = getLabelVector(QString::fromStdString(net2_label+(gt_label_list.at(i).toLocal8Bit().constData())));
        vector<pair<int, float>> net1_ious = this->calcIOUwithNetandGT(vecGT, vecNet1);  
        vector<pair<int, float>> net2_ious = this->calcIOUwithNetandGT(vecGT, vecNet2);
        net1_avg_ious.push_back((getAverageIOU(net1_ious))*100);
        net2_avg_ious.push_back((getAverageIOU(net2_ious))*100); 
        this->accTPFP(1, net1_ious);
        this->accTPFP(2, net2_ious);
    }

    vector<pair<QString, float>>cls_AP1 = this->calcAPbyClass(cls_pr1);
    vector<pair<QString, float>>cls_AP2 = this->calcAPbyClass(cls_pr2);

    int net1_cls_num = 0;
    float net1_cls_ap_sum = 0.0;
    bool net1_ok = false;
    for(size_t i=0; i<cls_AP1.size(); i++){
        if(cls_AP1[i].second > 0.0){
            net1_ok = true;
            net1_cls_num++;
            net1_cls_ap_sum += cls_AP1[i].second;
        }
    }
    float mAP1; 
    if(net1_ok) mAP1 = (net1_cls_ap_sum/net1_cls_num)*100;
    else mAP1 = 0.0;

    int net2_cls_num = 0;
    float net2_cls_ap_sum = 0.0;
    bool net2_ok = false;
    for(size_t i=0; i<cls_AP2.size(); i++){
        if(cls_AP2[i].second>0.0){
            net2_ok = true;
            net2_cls_num++;
            net2_cls_ap_sum += cls_AP2[i].second;
        }
    }
    float mAP2; 
    if(net2_ok) mAP2 = (net2_cls_ap_sum/net2_cls_num)*100;
    else mAP2 = 0.0;

    emit sendmAPs(mAP1,mAP2);

    emit sendNetAPs(1, cls_AP1);
    emit sendNetAPs(2, cls_AP2);

    emit sendAvgIOUs(net1_avg_ious, net2_avg_ious);
}

int* Metric2DOD::calcBoxes(float cx, float cy, float w, float h, int disp_w, int disp_h){
    static int box[6];
    float lx = cx-(w/2);
    float ly = cy-(h/2);
    float rx = cx+(w/2);
    float ry = cy+(h/2);
    box[0] = int(w*disp_w);
    box[1] = int(h*disp_h);
    box[2] = int(lx*disp_w);
    box[3] = int(ly*disp_h);
    box[4] = int(rx*disp_w);
    box[5] = int(ry*disp_h);
    return box;
}  

float Metric2DOD::getAverageIOU(vector<pair<int, float>> vec){
    float sum = 0.0;
    float avgIOU = 0.0;

    if(vec[0].first != -1){
        for(size_t i=0; i<vec.size(); i++){
            sum = sum + vec[i].second;
        }
        if(sum<=0.0) avgIOU = 0.0;
        else avgIOU = sum/float(vec.size()) + 1e-9;
    }
    
    return avgIOU;
}

bool Metric2DOD::checkIndex(vector<int> checked, int target){
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

void Metric2DOD::initializeVecArray(vector<pair<float, float>>* vec_array){
    for(size_t i=0; i<vec_array->size(); i++){
        vec_array[i].push_back(make_pair(0.0, 0.0));
    }
}


Metric2DOD::~Metric2DOD(){
    delete [] cls_gt;
    delete [] cls_pr1;
    delete [] cls_pr2;
}

