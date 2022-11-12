#include "metric3DOD.h"

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
#include <cmath>

#include <jsoncpp/json/json.h>

Metric3DOD::Metric3DOD(QObject *parent) : QThread(parent)
{
}

void Metric3DOD::setPaths(string _dataset_path)
{
    gt_label = _dataset_path + "/gt/label/";
    net1_label = _dataset_path + "/net1/label/";
    net2_label = _dataset_path + "/net2/label/";
    QString gt_label_path = QString::fromStdString(gt_label);
    QString net1_label_path = QString::fromStdString(net1_label);
    QString net2_label_path = QString::fromStdString(net2_label);

    QDir gt_label_dir(gt_label_path);
    QDir net1_label_dir(net1_label_path);
    QDir net2_label_dir(net2_label_path);

    gt_label_list = gt_label_dir.entryList(QStringList() << "*.json");
    net1_label_list = net1_label_dir.entryList(QStringList() << "*.json");
    net2_label_list = net2_label_dir.entryList(QStringList() << "*.json");
}

pair<float, float> Metric3DOD::returnAvgIOU(vector<BBoxes::BBox3D> _gtVec, vector<BBoxes::BBox3D> _net1Vec, vector<BBoxes::BBox3D> _net2Vec)
{
    vector<Metric3DOD::IoUInfo> net1_ious = this->calcIOUwithNetandGT(_gtVec, _net1Vec);
    vector<Metric3DOD::IoUInfo> net2_ious = this->calcIOUwithNetandGT(_gtVec, _net2Vec);
    float net1_avg_iou = this->getAverageIOU(net1_ious);
    float net2_avg_iou = this->getAverageIOU(net2_ious);
    return make_pair(net1_avg_iou, net2_avg_iou);
}

vector<BBoxes::BBox3D> Metric3DOD::getLabelVector(int type, QString label_path)
{
    vector<BBoxes::BBox3D> vec;
    Json::Value pred;
    ifstream in(label_path.toStdString().c_str());
    if (in.is_open())
        in >> pred;
    volume = 200 * 200 * 100;
    if (type == 0)
    {
        for (int i = 0; i < int(pred.size()); i++)
        {
            BBoxes::BBox3D bboxes;
            bboxes.conf = 0.0;
            bboxes.cx = pred[i]["psr"]["position"]["x"].asFloat();
            bboxes.cy = pred[i]["psr"]["position"]["y"].asFloat();
            bboxes.cz = pred[i]["psr"]["position"]["z"].asFloat();
            bboxes.dx = pred[i]["psr"]["scale"]["x"].asFloat();
            bboxes.dy = pred[i]["psr"]["scale"]["y"].asFloat();
            bboxes.dz = pred[i]["psr"]["scale"]["z"].asFloat();
            bboxes.yaw = pred[i]["psr"]["rotation"]["z"].asFloat();
            if (pred[i]["obj_id"].size() == 0)
                bboxes.cls = this->getIdxbyCls(pred[i]["obj_type"].asString());
            else
                bboxes.cls = pred[i]["obj_id"].asInt();
            vec.push_back(bboxes);
        }
    }
    else
    {
        for (int i = 0; i < int(pred.size()); i++)
        {
            BBoxes::BBox3D bboxes;
            bboxes.conf = pred[i]["score"].asFloat();
            bboxes.cx = pred[i]["psr"]["position"]["x"].asFloat();
            bboxes.cy = pred[i]["psr"]["position"]["y"].asFloat();
            bboxes.cz = pred[i]["psr"]["position"]["z"].asFloat();
            bboxes.dx = pred[i]["psr"]["scale"]["x"].asFloat();
            bboxes.dy = pred[i]["psr"]["scale"]["y"].asFloat();
            bboxes.dz = pred[i]["psr"]["scale"]["z"].asFloat();
            bboxes.yaw = pred[i]["psr"]["rotation"]["z"].asFloat();
            if (pred[i]["obj_id"].size() == 0)
                bboxes.cls = this->getIdxbyCls(pred[i]["obj_type"].asString());
            else
                bboxes.cls = pred[i]["obj_id"].asInt();
            vec.push_back(bboxes);
        }
    }

    return vec;
}

int Metric3DOD::getIdxbyCls(string cls)
{
    int idx;
    for (int i = 0; i < class_cnt; i++)
    {
        if (class_list[i].toStdString() == cls)
        {
            idx = i;
            break;
        }
        else
            idx = 0;
    }
    return idx;
}

bool compare_conf(BBoxes::BBox3D a, BBoxes::BBox3D b) { return a.conf > b.conf; }

bool compare_iou2(pair<int, float> a, pair<int, float> b) { return a.second > b.second; }

vector<Metric3DOD::IoUInfo> Metric3DOD::calcIOUwithNetandGT(vector<BBoxes::BBox3D> gtVec, vector<BBoxes::BBox3D> netVec)
{
    vector<Metric3DOD::IoUInfo> ious;
    sort(netVec.begin(), netVec.end(), compare_conf);
    vector<int> checked_gts = {0};

    for (size_t i = 0; i < netVec.size(); i++)
    {
        vector<pair<int, float>> each_ious;
        for (size_t j = 0; j < gtVec.size(); j++)
        {
            each_ious.push_back(make_pair((int)j, this->calcIOU(gtVec[j], netVec[i])));
        }
        sort(each_ious.begin(), each_ious.end(), compare_iou2);
        float iou = 0.0;
        if (i == 0)
        {
            iou = each_ious[0].second;
            checked_gts.push_back(each_ious[0].first);
        }
        else
        {
            for (size_t k = 0; k < checked_gts.size(); k++)
            {
                if (each_ious[0].first != checked_gts[k])
                {
                    iou = each_ious[0].second;
                    checked_gts.push_back(each_ious[0].first);
                    break;
                }
            }
        }
        ious.push_back({netVec[i].cls, netVec[i].conf, iou});
    }
    return ious;
}

void Metric3DOD::orgByClass(int type, vector<Metric3DOD::IoUInfo> iou_infos)
{
    for (size_t i = 0; i < iou_infos.size(); i++)
    {
        if (type == 1)
            cls_iou_info1[iou_infos[i].cls].push_back(iou_infos[i]);
        else if (type == 2)
            cls_iou_info2[iou_infos[i].cls].push_back(iou_infos[i]);
    }
}

float Metric3DOD::calcIOU(BBoxes::BBox3D gt, BBoxes::BBox3D net)
{
    BBoxes::Corners3D gt_corner = clsBBoxes.calcCorners(gt);
    BBoxes::Corners3D net_corner = clsBBoxes.calcCorners(net);

    pair<float, float> x_plane = this->calcXY(0, gt_corner.cor5, gt_corner.cor0, net_corner.cor5, net_corner.cor0);
    pair<float, float> y_plane = this->calcXY(1, gt_corner.cor4, gt_corner.cor3, net_corner.cor4, net_corner.cor3);
    pair<float, float> z_plane = this->calcXY(2, gt_corner.cor6, gt_corner.cor4, net_corner.cor6, net_corner.cor4);

    float iou = 0.0;
    if (x_plane.second < 0 || y_plane.first < 0 || z_plane.first < 0)
    {
        iou = 0.0;
        return iou;
    }
    float volume_overlap = x_plane.second * y_plane.first * z_plane.first;
    float volume_gt = abs(gt.dx * gt.dy * gt.dz);
    float volume_net = abs(net.dx * net.dy * net.dz);
    float volume_combined = volume_gt + volume_net - volume_overlap;
    if (volume_combined < 0)
    {
        iou = 0.0;
        return iou;
    }
    iou = volume_overlap / (volume_combined + 1e-5);

    return iou;
}

pair<float, float> Metric3DOD::calcXY(int ax, BBoxes::Corner3D gtl, BBoxes::Corner3D gtr, BBoxes::Corner3D netl, BBoxes::Corner3D netr)
{
    float x1, y1, x2, y2, width, height;
    if (ax == 0)
    { // x axes plane
        x1 = max(gtl.y, netl.y);
        y1 = min(gtl.z, netl.z);
        x2 = min(gtr.y, netr.y);
        y2 = max(gtr.z, netr.z);
        width = x2 - x1;
        height = y1 - y2;
    }
    else if (ax == 1)
    { // y axes plane
        x1 = min(gtl.x, netl.x);
        y1 = min(gtl.z, netl.z);
        x2 = max(gtr.x, netr.x);
        y2 = max(gtr.z, netr.z);
        width = x1 - x2;
        height = y1 - y2;
    }
    else if (ax == 2)
    { // z axes plane
        x1 = max(gtl.y, netl.y);
        y1 = max(gtl.x, netr.x);
        x2 = min(gtr.y, netr.y);
        y2 = min(gtr.x, netr.x);
        width = x2 - x1;
        height = y2 - y1;
    }
    /*
    xplane -> x : y, y : z
        axis :  + y , + z
    yplane -> x : x, y : z
        axis : - x , + z
    zplane -> z : y, z : x
        axis : + y, - x
    */
    return make_pair(width, height);
}

bool compare_conf2(Metric3DOD::IoUInfo a, Metric3DOD::IoUInfo b) { return a.conf > b.conf; }

vector<pair<float, float>> *Metric3DOD::accTPFP(vector<Metric3DOD::IoUInfo> *iou_info_list, float threshold)
{
    vector<pair<float, float>> *cls_pr = new vector<pair<float, float>>[class_cnt];
    initializeVecArray(cls_pr);
    for (int i = 0; i < class_cnt; i++)
    {
        vector<Metric3DOD::IoUInfo> iou_infos = iou_info_list[i];
        sort(iou_infos.begin(), iou_infos.end(), compare_conf2);
        vector<pair<int, int>> cls_tpfp = vector<pair<int, int>>(class_cnt, make_pair(0, 0));
        for (size_t j = 0; j < iou_infos.size(); j++)
        {
            float precision, recall = 0.0;
            if (iou_infos[j].cls == -1)
                continue;
            if (iou_infos[j].iou <= threshold)
                cls_tpfp[iou_infos[j].cls].second++; // FP
            else
                cls_tpfp[iou_infos[j].cls].first++; // TP

            if (cls_tpfp[iou_infos[j].cls].first == 0)
            {
                precision = 0.0;
                recall = 0.0;
            }
            else
            {
                precision = cls_tpfp[iou_infos[j].cls].first / float((cls_tpfp[iou_infos[j].cls].first + cls_tpfp[iou_infos[j].cls].second) + 1e-16);
                recall = cls_tpfp[iou_infos[j].cls].first / float((cls_gt[iou_infos[j].cls]) + 1e-16);
            }
            cls_pr[iou_infos[j].cls].push_back(make_pair(precision, recall));
        }
    }
    return cls_pr;
}

vector<pair<QString, float>> Metric3DOD::calcAPbyClass(vector<pair<float, float>> *cls_prs)
{
    vector<pair<QString, float>> cls_ap(class_cnt, make_pair("-", 0.0));
    for (int i = 0; i < class_cnt; i++)
    {
        float total_pr = 0.0;
        float recall_index = 10.0;
        float max_pr = 0.0;
        while (true)
        {
            if (recall_index < 0.0)
                break;
            for (size_t j = 0; j < cls_prs[i].size(); j++)
            {
                if ((recall_index / 10.0) - 0.1 < cls_prs[i][j].second && cls_prs[i][j].second <= recall_index / 10.0)
                {
                    if (max_pr <= cls_prs[i][j].first)
                        max_pr = cls_prs[i][j].first;
                }
            }
            total_pr = total_pr + max_pr;
            recall_index = recall_index - 1;
        }
        float AP = 0.0;
        if (total_pr == 0)
            AP = 0.0;
        else
            AP = total_pr / (11 + 1e-9);
        cls_ap.push_back(make_pair(class_list[i], AP));
    }
    return cls_ap;
}

void Metric3DOD::calcMetrics()
{
    float all_gt_size = 0.0;
    cls_gt = new int[class_cnt]{0};
    cls_iou_info1 = new vector<Metric3DOD::IoUInfo>[class_cnt];
    cls_iou_info2 = new vector<Metric3DOD::IoUInfo>[class_cnt];

    vector<float> net1_avg_ious;
    vector<float> net2_avg_ious;

    obj_size_list = new int[5]{0};

    for (int i = 0; i < gt_label_list.size(); i++)
    {
        vector<BBoxes::BBox3D> vecGT = getLabelVector(0, QString::fromStdString(gt_label + (gt_label_list.at(i).toLocal8Bit().constData())));
        all_gt_size += float(vecGT.size());
        for (size_t j = 0; j < vecGT.size(); j++)
        {
            cls_gt[vecGT[j].cls] += 1;
        }
        this->getObjSizeList(vecGT);
    }

    for (int i = 0; i < gt_label_list.size(); i++)
    {
        vector<BBoxes::BBox3D> vecGT = getLabelVector(0, QString::fromStdString(gt_label + (gt_label_list.at(i).toLocal8Bit().constData())));
        vector<BBoxes::BBox3D> vecNet1 = getLabelVector(1, QString::fromStdString(net1_label + (net1_label_list.at(i).toLocal8Bit().constData())));
        vector<BBoxes::BBox3D> vecNet2 = getLabelVector(2, QString::fromStdString(net2_label + (net2_label_list.at(i).toLocal8Bit().constData())));

        vector<Metric3DOD::IoUInfo> net1_ious = this->calcIOUwithNetandGT(vecGT, vecNet1);
        vector<Metric3DOD::IoUInfo> net2_ious = this->calcIOUwithNetandGT(vecGT, vecNet2);
        net1_avg_ious.push_back((getAverageIOU(net1_ious)) * 100);
        net2_avg_ious.push_back((getAverageIOU(net2_ious)) * 100);
        this->orgByClass(1, net1_ious);
        this->orgByClass(2, net2_ious);
    }

    float threshold = 0.7;

    vector<pair<float, float>> *cls_pr1 = this->accTPFP(cls_iou_info1, threshold);
    vector<pair<float, float>> *cls_pr2 = this->accTPFP(cls_iou_info2, threshold);
    vector<pair<QString, float>> cls_AP1 = this->calcAPbyClass(cls_pr1);
    vector<pair<QString, float>> cls_AP2 = this->calcAPbyClass(cls_pr2);

    int net1_cls_num = 0;
    float net1_cls_ap_sum = 0.0;
    for (size_t i = 0; i < cls_AP1.size(); i++)
    {
        if (cls_AP1[i].second > 0.0)
        {
            net1_cls_num++;
            net1_cls_ap_sum += cls_AP1[i].second;
        }
    }
    float mAP1 = net1_cls_ap_sum != 0 ? (net1_cls_ap_sum / net1_cls_num) * 100 : 0.0;

    int net2_cls_num = 0;
    float net2_cls_ap_sum = 0.0;
    for (size_t i = 0; i < cls_AP2.size(); i++)
    {
        if (cls_AP2[i].second > 0.0)
        {
            net2_cls_num++;
            net2_cls_ap_sum += cls_AP2[i].second;
        }
    }
    float mAP2 = net2_cls_ap_sum != 0 ? (net2_cls_ap_sum / net2_cls_num) * 100 : 0.0;

    float obj_sim = this->calcObjSimilarity(cls_iou_info1, cls_iou_info2);
    float class_var = this->calcNormVariance(cls_gt, class_cnt, all_gt_size);
    float obj_size_var = this->calcNormVariance(obj_size_list, 5.0, all_gt_size);
    float net1_bbox_acc = this->calcBBoxAcc(net1_avg_ious);
    float net2_bbox_acc = this->calcBBoxAcc(net2_avg_ious);
    float avg_bbox_acc = (net1_bbox_acc + net2_bbox_acc) / 2.0;

    emit sendNetAPs(1, cls_AP1);
    emit sendNetAPs(2, cls_AP2);

    emit sendObjSim(obj_sim);
    emit sendVariance(class_var, obj_size_var);
    emit sendAvgIOUs(net1_avg_ious, net2_avg_ious);
    emit sendBBoxAcc(net1_bbox_acc, net2_bbox_acc, avg_bbox_acc);
    emit sendmAPs(mAP1, mAP2);

    delete[] cls_pr1;
    delete[] cls_pr2;
}

float Metric3DOD::getAverageIOU(vector<Metric3DOD::IoUInfo> vec)
{
    float sum = 0.0;
    float avgIOU = 0.0;
    for (size_t i = 0; i < vec.size(); i++)
    {
        sum = sum + vec[i].iou;
    }
    if (sum <= 0.0)
        avgIOU = 0.0;
    else
        avgIOU = sum / (float(vec.size()) + 1e-9);
    return avgIOU;
}

float Metric3DOD::calcNormVariance(int *_list, int cnt_factor, int norm_factor)
{
    // Calc Average
    float cnt_sum = 0.0;
    for (int i = 0; i < cnt_factor; i++)
    {
        cnt_sum += _list[i];
    }
    float avg = cnt_sum != 0 ? cnt_sum / float(cnt_factor) : 0.0;
    avg = avg != 0 ? avg / float(norm_factor) : 0.0;
    // Normalizae
    float *norm_list = new float[cnt_factor]{0.0};
    for (int i = 0; i < cnt_factor; i++)
    {
        norm_list[i] = _list[i] != 0 ? float(_list[i]) / float(norm_factor) : 0.0;
    }

    // cout << "avg:" << avg << endl;
    float dev_sum = 0.0;
    for (int i = 0; i < cnt_factor; i++)
    {
        float dev = norm_list[i] - avg;
        // cout << norm_list[i] << " " << dev << endl;
        dev_sum += pow(dev, 2);
    }
    float var = dev_sum != 0.0 ? float(dev_sum) / float(cnt_factor) : 0.0;
    float std_var = var != 0.0 ? sqrt(var) : 0.0;
    // cout << "var:" << var << " "<< "std_var:" << std_var << endl<< endl;
    return std_var;
}

float Metric3DOD::calcClassConfVar(vector<Metric3DOD::IoUInfo> info)
{
    int *class_conf_list = this->getClassConfList(info);
    return this->calcNormVariance(class_conf_list, 5.0, info.size());
}

float Metric3DOD::calcObjSimilarity(vector<Metric3DOD::IoUInfo> *info1, vector<Metric3DOD::IoUInfo> *info2)
{
    float sum_conf_var = 0.0;
    for (int i = 0; i < class_cnt; i++)
    {
        // TODO : Indicate object is not same ;;;;
        float cls_conf_var1 = this->calcClassConfVar(info1[i]);
        float cls_conf_var2 = this->calcClassConfVar(info2[i]);
        sum_conf_var += (cls_conf_var1 + cls_conf_var2) / 2.0;
    }
    float avg_sim = sum_conf_var != 0 ? sum_conf_var / float(class_cnt) : 0.0;
    return avg_sim;
}

float Metric3DOD::calcBBoxAcc(vector<float> net)
{
    float sum = 0.0;
    for (size_t i = 0; i < net.size(); i++)
    {
        sum += net[i];
    }
    float avg = sum != 0 ? sum / float(net.size()) : 0.0;
    return avg;
}

int Metric3DOD::calcBoxSizeIndicator(BBoxes::BBox3D gt)
{
    int size = gt.dx * gt.dy * gt.dz;
    float weight[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    int size_ind = 0;
    for (int i = 1; i < 6; i++)
    {
        if ((volume * weight[i - 1]) < size && size <= (volume * weight[i]))
        {
            size_ind = i;
        }
    }
    return size_ind - 1;
}

void Metric3DOD::getObjSizeList(vector<BBoxes::BBox3D> vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        int size_ind = this->calcBoxSizeIndicator(vec[i]);
        obj_size_list[size_ind]++;
    }
}

int *Metric3DOD::getClassConfList(vector<Metric3DOD::IoUInfo> info)
{
    int *conf_list = new int[5]{0};
    for (size_t i = 0; i < info.size(); i++)
    {
        int idx = 0;
        for (int j = 1; j < 6; j++)
        {
            if ((float(j - 1) * 2.0 / 10.0) < info[i].conf && info[i].conf <= ((float(j) * 2.0) / 10.0))
            {
                idx = j;
            }
        }
        if (idx != 0)
        {
            conf_list[idx - 1]++;
        }
    }
    return conf_list;
}

bool Metric3DOD::checkIndex(vector<int> checked, int target)
{
    bool fine = true;
    for (size_t i = 0; i < checked.size(); i++)
    {
        if (target == checked[i])
        {
            fine = false;
            break;
        }
        else
        {
            fine = true;
        }
    }
    return fine;
}

void Metric3DOD::initializeVecArray(vector<pair<float, float>> *vec_array)
{
    for (size_t i = 0; i < vec_array->size(); i++)
    {
        vec_array[i].push_back(make_pair(0.0, 0.0));
    }
}

Metric3DOD::~Metric3DOD()
{
    delete[] cls_gt;
    delete[] cls_iou_info1;
    delete[] cls_iou_info2;
}
