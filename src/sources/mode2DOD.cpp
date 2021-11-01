#include "mode2DOD.h"

#include <iostream>
#include <string>
#include <unistd.h>	

#include <QWidget>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <QMetaType>
#include <QtCharts/QChartView>

mode2DOD::mode2DOD(QObject *parent) : QThread(parent){
    met2D = new Metric2D(this);
}

void mode2DOD::setData(string _dataset_path, QString _data_path, QString _label_path){
    dataset_path = _dataset_path;
    data_path = _data_path;
    label_path = _label_path;
    
    met2D->setPaths(dataset_path);

    QDir gt_data_dir(data_path);
	QDir gt_label_dir(label_path);

	img_data_list = gt_data_dir.entryList(QStringList() << "*.png");
	label_data_list = gt_label_dir.entryList(QStringList() << "*.txt");
	dir_size = img_data_list.size();
    emit sendImgList(img_data_list);
    setClassName();
	now_data_index = 0;
	now_img_data_name = img_data_list.at(now_data_index).toLocal8Bit().constData();
	now_label_data_name = label_data_list.at(now_data_index).toLocal8Bit().constData();
	now_img_data_path = QString::fromStdString(data_path.toStdString()+now_img_data_name);
    setBoxes();
}

void mode2DOD::calcAccuracy(){
    met2D->class_list = class_list;
    met2D->threshold = float(this->threshold)/100;
    met2D->calcMetrics();
}

void mode2DOD::setBoxes(){
    met2D->now_img_data_path = now_img_data_path; 
	gt_label_path = QString::fromStdString(dataset_path+"/gt/label/"+now_label_data_name);
	vector<BBoxes::BBox2D> vecGT = met2D->getLabelVector(gt_label_path);
    drawBoxes(0, vecGT);
	net1_label_path = QString::fromStdString(dataset_path+"/net1/label/"+now_label_data_name);	
	vector<BBoxes::BBox2D> vecNet1 = met2D->getLabelVector(net1_label_path);
    drawBoxes(1, vecNet1);
	net2_label_path = QString::fromStdString(dataset_path+"/net2/label/"+now_label_data_name);
	vector<BBoxes::BBox2D> vecNet2 = met2D->getLabelVector(net2_label_path);
    drawBoxes(2, vecNet2);
    pair<float, float> avg_iou = met2D->returnAvgIOU(vecGT, vecNet1, vecNet2);
    emit sendAvgIOU((avg_iou.first)*100, (avg_iou.second)*100);
}


void mode2DOD::drawBoxes(int type, vector<BBoxes::BBox2D> bboxes){
    QImage target_img(now_img_data_path);
    QPainter *painter = new QPainter(&target_img);
    QFont f("Helvetica [Cronyx]", 30, QFont::Bold); 

    for(size_t i=0; i<bboxes.size(); i++) {
        QPen pen = this->getBBoxPen(bboxes[i].cls);        
        painter->setPen(pen);
        QRect qrect = QRect(bboxes[i].lx,bboxes[i].ly, bboxes[i].w, bboxes[i].h);
        painter->drawRect(qrect);
        painter->setFont(f);
        QString class_name = class_list.at(bboxes[i].cls);
        painter->drawText(bboxes[i].lx, bboxes[i].ly-10, class_name);
    }
    if(type==0){
        emit sendGTImg(target_img);
    }else if(type == 1){
        emit sendNet1Img(target_img);
    }else if(type==2){
        emit sendNet2Img(target_img);
    }
    painter->end();
}

void mode2DOD::setThreshold(int _th){
    this->threshold = _th;
    met2D->threshold = float(this->threshold)/100;
}

void mode2DOD::setDataIdx(int idx){
    now_img_data_name = img_data_list.at(idx).toLocal8Bit().constData();
	now_img_data_path = QString::fromStdString(data_path.toStdString()+now_img_data_name);
	now_label_data_name = label_data_list.at(idx).toLocal8Bit().constData();
	setBoxes();
}

void mode2DOD::goLeft(){
	if (now_data_index > 0) now_data_index -= 1;
	else now_data_index = 0;
	now_img_data_name = img_data_list.at(now_data_index).toLocal8Bit().constData();
	now_img_data_path = QString::fromStdString(data_path.toStdString()+now_img_data_name);
	now_label_data_name = label_data_list.at(now_data_index).toLocal8Bit().constData();
	setBoxes();
}
void mode2DOD::goRight(){
	if (now_data_index < dir_size-1) now_data_index += 1;
	else now_data_index = dir_size-1;
	now_img_data_name = img_data_list.at(now_data_index).toLocal8Bit().constData();
	now_img_data_path = QString::fromStdString(data_path.toStdString()+now_img_data_name);	
	now_label_data_name = label_data_list.at(now_data_index).toLocal8Bit().constData();
	setBoxes();
}

void mode2DOD::setClassName(){
    string class_path = dataset_path+"/classes.txt";
    QFile class_file(QString::fromStdString(class_path));
    if(!class_file.open(QIODevice::ReadOnly)){
        cout<<"error to open file ["+dataset_path+"/classes.txt ]."<<endl;
    }
	QTextStream in(&class_file);

    class_cnt=0;
    while(!in.atEnd()){
        QString line = in.readLine();    
        class_list << line;
        class_cnt ++;
    }
    met2D->class_cnt = class_cnt;
}

QPen mode2DOD::getBBoxPen(int class_num){
    QPen pen;
    pen.setWidth(5);
    if (class_num % 4 == 0){
        QBrush qb("#9580FF");
        pen.setBrush(qb);
    }
    else if(class_num % 4 == 1){
        QBrush qb("#7FFDEA");
        pen.setBrush(qb);
    }else if(class_num % 4 == 2){
        QBrush qb("#89FD80");
        pen.setBrush(qb);
    }else if(class_num % 4 == 3){
        QBrush qb("#FFFF80");
        pen.setBrush(qb);
    }
    else{
        QBrush qb("#F47FBF");
        pen.setBrush(qb);
    }
    return pen;
}