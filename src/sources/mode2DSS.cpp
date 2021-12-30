#include "mode2DSS.h"

#include <iostream>
#include <string>
#include <unistd.h>	
#include <fstream>

#include <QWidget>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <QMetaType>
#include <QtCharts/QChartView>
#include <QCoreApplication>

#include <jsoncpp/json/json.h>
 
mode2DSS::mode2DSS(QObject *parent) : QThread(parent){
    met2DSS = new Metric2DSS(this);
}


void mode2DSS::setData(string _dataset_path){
    dataset_path = _dataset_path;
    gt_data_path = QString::fromStdString(dataset_path+"/gt/data/"); 
    net1_data_path = QString::fromStdString(dataset_path+"/net1/data/"); 
    net2_data_path = QString::fromStdString(dataset_path+"/net2/data/"); 
    
    QDir gt_data_dir(gt_data_path);

	img_data_list = gt_data_dir.entryList(QStringList() << "*.png" << "*.jpg");
	dir_size = img_data_list.size();
    emit sendImgList(img_data_list);
    met2DSS->setPaths(img_data_list, gt_data_path, net1_data_path, net2_data_path);

    setClassList();
	now_data_index = 0;
	now_img_data_name = img_data_list.at(now_data_index).toLocal8Bit().constData();
	gt_now_img_data_path = QString::fromStdString(gt_data_path.toStdString()+now_img_data_name);
    net1_now_img_data_path = QString::fromStdString(net1_data_path.toStdString()+now_img_data_name);
    net2_now_img_data_path = QString::fromStdString(net2_data_path.toStdString()+now_img_data_name);
    met2DSS->setClassVectors();
    setPolygons();
}

void mode2DSS::saveAccept(string storage_path){
    QFile::copy(gt_now_img_data_path,QString::fromStdString(storage_path+"/accept/gt/data/"+now_img_data_name)); 
    QFile::copy(net1_now_img_data_path, QString::fromStdString(storage_path+"/accept/net1/data/"+now_img_data_name));
    QFile::copy(net2_now_img_data_path, QString::fromStdString(storage_path+"/accept/net2/data/"+now_img_data_name));
}

void mode2DSS::saveReject(string storage_path){
    QFile::copy(gt_now_img_data_path,QString::fromStdString(storage_path+"/reject/gt/data/"+now_img_data_name)); 
    QFile::copy(net1_now_img_data_path, QString::fromStdString(storage_path+"/reject/net1/data/"+now_img_data_name));
    QFile::copy(net2_now_img_data_path, QString::fromStdString(storage_path+"/reject/net2/data/"+now_img_data_name));
}


void mode2DSS::setPolygons(){
	QImage gt_img(gt_now_img_data_path);
    emit sendGTImg(gt_img);
    QImage net1_img(net1_now_img_data_path);
    emit sendNet1Img(net1_img);
    QImage net2_img(net2_now_img_data_path);
    emit sendNet2Img(net2_img);
    met2DSS->calcMetricbyFrame(gt_img, net1_img, net2_img);
}

void mode2DSS::setClassList(){
    string class_path = dataset_path+"/classes.json";
    Json::Value classes;
    ifstream in(class_path.c_str());
    if(in.is_open()) in>>classes;
    for(int i=0; i<int(classes.size()); i++){
        string name = classes[i]["name"].asString();
        string color = classes[i]["color"].asString();
        class_list.push_back(make_pair(name, color));
    }
    met2DSS->class_cnt = classes.size();
    met2DSS->class_list = class_list;
}

void mode2DSS::calcAccuracy(){
    met2DSS->threshold = float(this->threshold)/100;
    emit sendStart();
    QCoreApplication::processEvents();
    met2DSS->calcMetrics();
    emit sendStop();
}


void mode2DSS::setThreshold(int _th){
    this->threshold = _th;
}

void mode2DSS::setDataIdx(int idx){
    now_img_data_name = img_data_list.at(idx).toLocal8Bit().constData();
	gt_now_img_data_path = QString::fromStdString(gt_data_path.toStdString()+now_img_data_name);
    net1_now_img_data_path = QString::fromStdString(net1_data_path.toStdString()+now_img_data_name);
    net2_now_img_data_path = QString::fromStdString(net2_data_path.toStdString()+now_img_data_name);
	setPolygons();
}

void mode2DSS::goLeft(){
	if (now_data_index > 0) now_data_index -= 1;
	else now_data_index = 0;
	now_img_data_name = img_data_list.at(now_data_index).toLocal8Bit().constData();
	gt_now_img_data_path = QString::fromStdString(gt_data_path.toStdString()+now_img_data_name);
    net1_now_img_data_path = QString::fromStdString(net1_data_path.toStdString()+now_img_data_name);
    net2_now_img_data_path = QString::fromStdString(net2_data_path.toStdString()+now_img_data_name);
	setPolygons();
}
void mode2DSS::goRight(){
	if (now_data_index < dir_size-1) now_data_index += 1;
	else now_data_index = dir_size-1;
	now_img_data_name = img_data_list.at(now_data_index).toLocal8Bit().constData();
	gt_now_img_data_path = QString::fromStdString(gt_data_path.toStdString()+now_img_data_name);
    net1_now_img_data_path = QString::fromStdString(net1_data_path.toStdString()+now_img_data_name);
    net2_now_img_data_path = QString::fromStdString(net2_data_path.toStdString()+now_img_data_name);
	setPolygons();
}