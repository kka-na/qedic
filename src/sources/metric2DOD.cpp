#include "metric2DOD.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>

#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QImage>

Metric2DOD::Metric2DOD(QObject *parent) : QThread(parent)
{
}

void Metric2DOD::setPaths(string _dataset_path)
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

    gt_label_list = gt_label_dir.entryList(QStringList() << "*.txt");
    net1_label_list = net1_label_dir.entryList(QStringList() << "*.txt");
    net2_label_list = net2_label_dir.entryList(QStringList() << "*.txt");
}

pair<float, float> Metric2DOD::returnAvgIOU(vector<BBoxes::BBox2D> _gtVec, vector<BBoxes::BBox2D> _net1Vec, vector<BBoxes::BBox2D> _net2Vec)
{
    vector<Metric2DOD::IoUInfo> net1_ious = this->calcIOUwithNetandGT(_gtVec, _net1Vec);
    vector<Metric2DOD::IoUInfo> net2_ious = this->calcIOUwithNetandGT(_gtVec, _net2Vec);
    float net1_avg_iou = this->getAverageIOU(net1_ious);
    float net2_avg_iou = this->getAverageIOU(net2_ious);
    return make_pair(net1_avg_iou, net2_avg_iou);
}

vector<BBoxes::BBox2D> Metric2DOD::getLabelVector(QString label_path)
{
    vector<BBoxes::BBox2D> vec;
    QFile label_file(label_path);
    if (!label_file.open(QIODevice::ReadOnly))
    {
        cout << "The [" + label_path.toStdString() + "] was not detected." << endl;
        BBoxes::BBox2D init = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0};
        vec.push_back(init);
    }
    else
    {
        QTextStream in(&label_file);

        QImage target_img(now_img_data_path);
        int img_w = target_img.width();
        int img_h = target_img.height();
        volume = img_w * img_h;
        while (!in.atEnd())
        {
            BBoxes::BBox2D bboxes;
            QString line = in.readLine();
            QStringList fields = line.split(" ");

            bboxes.cls = fields.at(0).toInt();
            float cx, cy, w, h;
            if (fields.size() == 5)
            { // no confidence, this is GT
                bboxes.conf = 1.0;
                cx = fields.at(1).toFloat();
                cy = fields.at(2).toFloat();
                w = fields.at(3).toFloat();
                h = fields.at(4).toFloat();
            }
            else
            {
                bboxes.conf = fields.at(1).toFloat();
                cx = fields.at(2).toFloat();
                cy = fields.at(3).toFloat();
                w = fields.at(4).toFloat();
                h = fields.at(5).toFloat();
            }
            int *box = calcBoxes(cx, cy, w, h, img_w, img_h);

            bboxes.w = box[0];
            bboxes.h = box[1];
            bboxes.lx = box[2];
            bboxes.ly = box[3];
            bboxes.rx = box[4];
            bboxes.ry = box[5];
            bboxes.cx = int(cx * img_w);
            bboxes.cy = int(cy * img_h);
            vec.push_back(bboxes);
        }
    }
    label_file.close();
    return vec;
}

float Metric2DOD::calcIOU(BBoxes::BBox2D gt, BBoxes::BBox2D net)
{
    int x1 = max(gt.lx, net.lx);
    int y1 = max(gt.ly, net.ly);
    int x2 = min(gt.rx, net.rx);
    int y2 = min(gt.ry, net.ry);
    int tmpwidth = (x2 - x1);
    int tmpheight = (y2 - y1);

    float iou = 0;
    if (tmpwidth < 0 || tmpheight < 0)
    {
        iou = 0.0;
        return iou;
    }
    int area_overlap = tmpwidth * tmpheight;
    int area_a = (gt.rx - gt.lx) * (gt.ry - gt.ly);
    int area_b = (net.rx - net.lx) * (net.ry - net.ly);
    int area_combined = area_a + area_b - area_overlap;

    iou = float(area_overlap / (area_combined + 1e-9));

    return iou;
}

bool compare_conf(BBoxes::BBox2D a, BBoxes::BBox2D b) { return a.conf > b.conf; }

bool compare_iou(pair<int, float> a, pair<int, float> b) { return a.second > b.second; }

vector<Metric2DOD::IoUInfo> Metric2DOD::calcIOUwithNetandGT(vector<BBoxes::BBox2D> gtVec, vector<BBoxes::BBox2D> netVec)
{
    vector<Metric2DOD::IoUInfo> iouInfos;
    sort(netVec.begin(), netVec.end(), compare_conf);
    vector<int> checked_gts = {0};
    bool detected = false;
    for (size_t i = 0; i < netVec.size(); i++)
    {
        if (netVec[i].cls == -1)
            continue;
        detected = true;
        vector<pair<int, float>> each_ious;
        for (size_t j = 0; j < gtVec.size(); j++)
        {
            each_ious.push_back(make_pair((int)j, this->calcIOU(gtVec[j], netVec[i])));
        }
        sort(each_ious.begin(), each_ious.end(), compare_iou);
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
        iouInfos.push_back({netVec[i].cls, netVec[i].conf, iou});
    }
    if (!detected)
        iouInfos.push_back({-1, 0.0, 0.0});
    return iouInfos;
}

void Metric2DOD::orgByClass(int type, vector<Metric2DOD::IoUInfo> iou_infos)
{
    for (size_t i = 0; i < iou_infos.size(); i++)
    {
        if (iou_infos[i].cls == -1)
            return;
        else
        {
            if (type == 1)
                cls_iou_info1[iou_infos[i].cls].push_back(iou_infos[i]);
            else if (type == 2)
                cls_iou_info2[iou_infos[i].cls].push_back(iou_infos[i]);
        }
    }
}

bool compare_conf2(Metric2DOD::IoUInfo a, Metric2DOD::IoUInfo b) { return a.conf > b.conf; }

vector<pair<float, float>> *Metric2DOD::accTPFP(vector<Metric2DOD::IoUInfo> *iou_info_list, float threshold)
{

    vector<pair<float, float>> *cls_pr = new vector<pair<float, float>>[class_cnt]; // = {make_pair(0.0, 0.0)};
    initializeVecArray(cls_pr);
    for (int i = 0; i < class_cnt; i++)
    {
        vector<Metric2DOD::IoUInfo> iou_infos = iou_info_list[i];
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

// COCO Method: mAP Calculating by 101-point iterpolation
vector<pair<QString, float>> Metric2DOD::calcAPbyCOCO(vector<pair<float, float>> *cls_prs)
{
    vector<pair<QString, float>> cls_ap(class_cnt, make_pair("-", 0.0));
    for (int i = 0; i < class_cnt; i++)
    {
        float total_pr = 0.0;
        float recall_index = 100.0;
        float max_pr = 0.0;

        while (true)
        {
            if (recall_index < 0.00)
                break;
            for (size_t j = 0; j < cls_prs[i].size(); j++)
            {
                if ((recall_index / 100.0) - 0.01 < cls_prs[i][j].second && cls_prs[i][j].second <= recall_index / 100.0)
                {
                    if (max_pr <= cls_prs[i][j].first)
                        max_pr = cls_prs[i][j].first;
                }
            }
            total_pr += max_pr;
            recall_index = recall_index - 1;
        }

        float AP = 0.0;
        if (total_pr == 0)
            AP = 0.0;
        else
            AP = float(total_pr / (101.0 + 1e-16));
        cls_ap.push_back(make_pair(class_list[i], AP));
    }
    return cls_ap;
}

// VOC Pascal Method : Every point interpolation
vector<pair<QString, float>> Metric2DOD::calcAPbyVOC(vector<pair<float, float>> *cls_prs)
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
            total_pr += max_pr;
            recall_index = recall_index - 1;
        }
        float AP = 0.0;
        if (total_pr == 0)
            AP = 0.0;
        else
            AP = float(total_pr / (11 + 1e-16));
        cls_ap.push_back(make_pair(class_list[i], AP));
    }
    return cls_ap;
}

void Metric2DOD::calcMetrics()
{
    float all_gt_size = 0.0;
    cls_gt = new int[class_cnt]{0};
    cls_iou_info1 = new vector<Metric2DOD::IoUInfo>[class_cnt];
    cls_iou_info2 = new vector<Metric2DOD::IoUInfo>[class_cnt];
    vector<float> net1_avg_ious;
    vector<float> net2_avg_ious;
    obj_size_list = new int[5]{0};

    for (int i = 0; i < gt_label_list.size(); i++)
    {
        vector<BBoxes::BBox2D> vecGT = getLabelVector(QString::fromStdString(gt_label + (gt_label_list.at(i).toLocal8Bit().constData())));
        all_gt_size += float(vecGT.size());
        for (size_t j = 0; j < vecGT.size(); j++)
        {
            cls_gt[vecGT[j].cls] += 1;
        }
        this->getObjSizeList(vecGT);
    }
    for (int i = 0; i < gt_label_list.size(); i++)
    {
        vector<BBoxes::BBox2D> vecGT = getLabelVector(QString::fromStdString(gt_label + (gt_label_list.at(i).toLocal8Bit().constData())));
        vector<BBoxes::BBox2D> vecNet1 = getLabelVector(QString::fromStdString(net1_label + (gt_label_list.at(i).toLocal8Bit().constData())));
        vector<BBoxes::BBox2D> vecNet2 = getLabelVector(QString::fromStdString(net2_label + (gt_label_list.at(i).toLocal8Bit().constData())));
        vector<Metric2DOD::IoUInfo> net1_ious = this->calcIOUwithNetandGT(vecGT, vecNet1);
        vector<Metric2DOD::IoUInfo> net2_ious = this->calcIOUwithNetandGT(vecGT, vecNet2);
        net1_avg_ious.push_back((getAverageIOU(net1_ious)) * 100);
        net2_avg_ious.push_back((getAverageIOU(net2_ious)) * 100);
        this->orgByClass(1, net1_ious);
        this->orgByClass(2, net2_ious);
    }

    float threshold = 0.5;
    float sum_mAP1 = 0.0;
    float sum_mAP2 = 0.0;
    for (int n = 0; n < 10; n++)
    {
        vector<pair<float, float>> *cls_pr1 = this->accTPFP(cls_iou_info1, threshold);
        vector<pair<float, float>> *cls_pr2 = this->accTPFP(cls_iou_info2, threshold);
        vector<pair<QString, float>> cls_AP1 = this->calcAPbyVOC(cls_pr1);
        vector<pair<QString, float>> cls_AP2 = this->calcAPbyVOC(cls_pr2);

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

        sum_mAP1 += mAP1;
        sum_mAP2 += mAP2;
        threshold += 0.05;
        if (n == 0)
        {
            emit sendNetAPs(1, cls_AP1);
            emit sendNetAPs(2, cls_AP2);
        }
        delete[] cls_pr1;
        delete[] cls_pr2;
    }
    float obj_sim = this->calcObjSimilarity(cls_iou_info1, cls_iou_info2);
    float class_var = this->calcNormVariance(cls_gt, class_cnt, all_gt_size);
    float obj_size_var = this->calcNormVariance(obj_size_list, 5.0, all_gt_size);
    float net1_bbox_acc = this->calcBBoxAcc(net1_avg_ious);
    float net2_bbox_acc = this->calcBBoxAcc(net2_avg_ious);
    float avg_bbox_acc = (net1_bbox_acc + net2_bbox_acc) / 2.0;
    float net1_mAP = float(sum_mAP1 / 10 + 1e-16);
    float net2_mAP = float(sum_mAP2 / 10 + 1e-16);

    emit sendObjSim(obj_sim);
    emit sendVariance(class_var, obj_size_var);
    emit sendAvgIOUs(net1_avg_ious, net2_avg_ious);
    emit sendBBoxAcc(net1_bbox_acc, net2_bbox_acc, avg_bbox_acc);
    emit sendmAPs(net1_mAP, net2_mAP);
}

float Metric2DOD::calcNormVariance(int *_list, int cnt_factor, int norm_factor)
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

float Metric2DOD::calcClassConfVar(vector<Metric2DOD::IoUInfo> info)
{
    int *class_conf_list = this->getClassConfList(info);
    return this->calcNormVariance(class_conf_list, 5.0, info.size());
}

float Metric2DOD::calcObjSimilarity(vector<Metric2DOD::IoUInfo> *info1, vector<Metric2DOD::IoUInfo> *info2)
{
    float sum_conf_var = 0.0;
    for (int i = 0; i < class_cnt; i++)
    {
        float cls_conf_var1 = this->calcClassConfVar(info1[i]);
        float cls_conf_var2 = this->calcClassConfVar(info2[i]);
        sum_conf_var += (cls_conf_var1 + cls_conf_var2) / 2.0;
    }
    float avg_sim = sum_conf_var / float(class_cnt);
    return avg_sim;
}

float Metric2DOD::calcBBoxAcc(vector<float> net)
{
    float sum = 0.0;
    for (size_t i = 0; i < net.size(); i++)
    {
        sum += net[i];
    }
    float avg = sum / float(net.size());
    return avg;
}

int Metric2DOD::calcBoxSizeIndicator(BBoxes::BBox2D gt)
{
    int size = gt.w * gt.h;
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

void Metric2DOD::getObjSizeList(vector<BBoxes::BBox2D> vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        int size_ind = this->calcBoxSizeIndicator(vec[i]);
        obj_size_list[size_ind]++;
    }
}

int *Metric2DOD::getClassConfList(vector<Metric2DOD::IoUInfo> info)
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

int *Metric2DOD::calcBoxes(float cx, float cy, float w, float h, int disp_w, int disp_h)
{
    static int box[6];
    float lx = cx - (w / 2);
    float ly = cy - (h / 2);
    float rx = cx + (w / 2);
    float ry = cy + (h / 2);
    box[0] = int(w * disp_w);
    box[1] = int(h * disp_h);
    box[2] = int(lx * disp_w);
    box[3] = int(ly * disp_h);
    box[4] = int(rx * disp_w);
    box[5] = int(ry * disp_h);
    return box;
}

float Metric2DOD::getAverageIOU(vector<Metric2DOD::IoUInfo> vec)
{
    float sum = 0.0;
    float avgIOU = 0.0;

    if (vec[0].cls != -1)
    {
        for (size_t i = 0; i < vec.size(); i++)
        {
            sum = sum + vec[i].iou;
        }
        if (sum <= 0.0)
            avgIOU = 0.0;
        else
            avgIOU = sum / float(vec.size()) + 1e-9;
    }

    return avgIOU;
}

bool Metric2DOD::checkIndex(vector<int> checked, int target)
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

void Metric2DOD::initializeVecArray(vector<pair<float, float>> *vec_array)
{
    for (size_t i = 0; i < vec_array->size(); i++)
    {
        vec_array[i].push_back(make_pair(0.0, 0.0));
    }
}

Metric2DOD::~Metric2DOD()
{
    delete[] cls_gt;
    delete[] obj_size_list;
    delete[] cls_iou_info1;
    delete[] cls_iou_info2;
}
