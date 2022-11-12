#include "mode3DOD.h"

#include <iostream>
#include <string>
#include <unistd.h>

#include <QWidget>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <QMetaType>
#include <QtCharts/QChartView>

mode3DOD::mode3DOD(QObject *parent) : QThread(parent)
{
    met3DOD = new Metric3DOD(this);
}

void mode3DOD::setData(string _dataset_path, QString _data_path, QString _label_path)
{
    dataset_path = _dataset_path;
    data_path = _data_path;
    label_path = _label_path;

    met3DOD->setPaths(dataset_path);

    QDir gt_data_dir(data_path);
    QDir gt_label_dir(label_path);

    pcd_data_list = gt_data_dir.entryList(QStringList() << "*.pcd");
    label_data_list = gt_label_dir.entryList(QStringList() << "*.json");
    dir_size = pcd_data_list.size();
    emit sendPCDList(pcd_data_list);
    setClassName();
    now_data_index = 0;
    now_pcd_data_name = pcd_data_list.at(now_data_index).toLocal8Bit().constData();
    now_label_data_name = label_data_list.at(now_data_index).toLocal8Bit().constData();
    now_pcd_data_path = QString::fromStdString(data_path.toStdString() + now_pcd_data_name);
    setBoxes();
}

void mode3DOD::saveAccept(string storage_path)
{
    QFile::copy(now_pcd_data_path, QString::fromStdString(storage_path + "/accept/gt/data/" + now_pcd_data_name));
    QFile::copy(gt_label_path, QString::fromStdString(storage_path + "/accept/gt/label/" + now_label_data_name));
    QFile::copy(net1_label_path, QString::fromStdString(storage_path + "/accept/net1/label/" + now_label_data_name));
    QFile::copy(net2_label_path, QString::fromStdString(storage_path + "/accept/net2/label/" + now_label_data_name));
}

void mode3DOD::saveReject(string storage_path)
{
    QFile::copy(now_pcd_data_path, QString::fromStdString(storage_path + "/reject/gt/data/" + now_pcd_data_name));
    QFile::copy(gt_label_path, QString::fromStdString(storage_path + "/reject/gt/label/" + now_label_data_name));
    QFile::copy(net1_label_path, QString::fromStdString(storage_path + "/reject/net1/label/" + now_label_data_name));
    QFile::copy(net2_label_path, QString::fromStdString(storage_path + "/reject/net2/label/" + now_label_data_name));
}

void mode3DOD::setClassName()
{
    string class_path = dataset_path + "/classes.txt";
    QFile class_file(QString::fromStdString(class_path));
    if (!class_file.open(QIODevice::ReadOnly))
    {
        cout << "error to open file [" + dataset_path + "/classes.txt ]." << endl;
    }
    QTextStream in(&class_file);

    class_cnt = 0;
    while (!in.atEnd())
    {
        QString line = in.readLine();
        class_list << line;
        class_cnt++;
    }
    met3DOD->class_cnt = class_cnt;
    met3DOD->class_list = class_list;
}

void mode3DOD::setBoxes()
{
    met3DOD->now_pcd_data_path = now_pcd_data_path;
    gt_label_path = QString::fromStdString(dataset_path + "/gt/label/" + now_label_data_name);
    vector<BBoxes::BBox3D> vecGT = met3DOD->getLabelVector(0, gt_label_path);
    emit sendGTPCD(now_pcd_data_path, vecGT);
    net1_label_path = QString::fromStdString(dataset_path + "/net1/label/" + now_label_data_name);
    vector<BBoxes::BBox3D> vecNet1 = met3DOD->getLabelVector(1, net1_label_path);
    emit sendNet1PCD(now_pcd_data_path, vecNet1);
    net2_label_path = QString::fromStdString(dataset_path + "/net2/label/" + now_label_data_name);
    vector<BBoxes::BBox3D> vecNet2 = met3DOD->getLabelVector(2, net2_label_path);
    emit sendNet2PCD(now_pcd_data_path, vecNet2);
    pair<float, float> avg_iou = met3DOD->returnAvgIOU(vecGT, vecNet1, vecNet2);
    emit sendAvgIOU((avg_iou.first) * 100, (avg_iou.second) * 100);
}

void mode3DOD::calcAccuracy()
{
    // met3DOD->threshold = float(this->threshold) / 100;
    met3DOD->calcMetrics();
}

void mode3DOD::setThreshold(int _th)
{
    this->threshold = _th;
}

void mode3DOD::setDataIdx(int idx)
{
    now_pcd_data_name = pcd_data_list.at(idx).toLocal8Bit().constData();
    now_pcd_data_path = QString::fromStdString(data_path.toStdString() + now_pcd_data_name);
    now_label_data_name = label_data_list.at(idx).toLocal8Bit().constData();
    setBoxes();
}

void mode3DOD::goLeft()
{
    if (now_data_index > 0)
        now_data_index -= 1;
    else
        now_data_index = 0;
    now_pcd_data_name = pcd_data_list.at(now_data_index).toLocal8Bit().constData();
    now_pcd_data_path = QString::fromStdString(data_path.toStdString() + now_pcd_data_name);
    now_label_data_name = label_data_list.at(now_data_index).toLocal8Bit().constData();
    setBoxes();
}
void mode3DOD::goRight()
{
    if (now_data_index < dir_size - 1)
        now_data_index += 1;
    else
        now_data_index = dir_size - 1;
    now_pcd_data_name = pcd_data_list.at(now_data_index).toLocal8Bit().constData();
    now_pcd_data_path = QString::fromStdString(data_path.toStdString() + now_pcd_data_name);
    now_label_data_name = label_data_list.at(now_data_index).toLocal8Bit().constData();
    setBoxes();
}
