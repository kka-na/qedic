#include "mainwindow.h"

#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <unistd.h>	
#include <algorithm> 

#include <QCoreApplication>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QMetaType>
#include <QStorageInfo>
#include <QtCharts>
 
#include "jsoncpp/json/json.h"

#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindowInteractor.h>

MainWindow::MainWindow(QWidget *parent):QMainWindow(parent),ui(new Ui::MainWindow){
	ui->setupUi(this);
	this->setFunction();
}

MainWindow::~MainWindow(){
	delete ui;
}

void MainWindow::setFunction(){
	connect(ui->taskButton, SIGNAL(clicked()), this, SLOT(setTask()));
	connect(ui->serverButton, SIGNAL(clicked()), this, SLOT(setServer()));
	connect(ui->dataButton, SIGNAL(clicked()), this, SLOT(setData()));
	connect(ui->storageButton, SIGNAL(clicked()), this, SLOT(setStorage()));
	connect(ui->detailButton, SIGNAL(clicked()), this, SLOT(displayDetail()));
	connect(ui->acceptButton, SIGNAL(clicked()), this, SLOT(dataAccept()));
	connect(ui->rejectButton, SIGNAL(clicked()), this, SLOT(dataReject()));
	connect(ui->thSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setThreshold(int)));
	connect(ui->dataList, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(setListIdx(QListWidgetItem*)));
}

void MainWindow::setTask(){
	QInputDialog qDialog;
	QStringList items;
	items << QString("2D Object Detection"); items << QString("2D Semantic Segmentation"); items << QString("3D Object Detection");
	qDialog.setOptions(QInputDialog::UseListViewForComboBoxItems);
	qDialog.setStyleSheet("QInputDialog {background-color: #F1F5F8;}");
	qDialog.setComboBoxItems(items);
	qDialog.setWindowTitle("Choose Server");
	if(qDialog.exec()){
		task = qDialog.textValue().toStdString();
		task_idx = qDialog.comboBoxItems().indexOf(qDialog.textValue());
	}
	this->setEachTasks();
}

void MainWindow::setEachTasks(){
	this->clearLayouts();
	threshold = ui->thSpinBox->value();
	if(task_idx == 0){
		ui->net1Name->setText(" Scaled YOLOv4 - P7"); net1_achieve = 55.5;
		ui->net2Name->setText(" CenterNet 2"); net2_achieve = 45.6;
		od2d = new mode2DOD(this);
		od2d->threshold = threshold;
		this->set2DODLayouts();
		connect(od2d, SIGNAL(sendImgList(QStringList)), this, SLOT(setImageList(QStringList)));
		connect(od2d, SIGNAL(sendGTImg(QImage)), this, SLOT(setGTImage(QImage)));
		connect(od2d, SIGNAL(sendNet1Img(QImage)), this, SLOT(setNet1Image(QImage)));
		connect(od2d, SIGNAL(sendNet2Img(QImage)), this, SLOT(setNet2Image(QImage)));
		connect(od2d, SIGNAL(sendStart()), this, SLOT(setLoadingMovie()));
		connect(od2d, SIGNAL(sendStop()), this, SLOT(stopLoadingMovie()));
		connect(od2d, SIGNAL(sendAvgIOU(float, float)), this, SLOT(setAvgIOU(float, float)));
		connect(od2d->met2DOD, SIGNAL(sendmAPs(float, float)), this, SLOT(setmAPs(float, float)));
		connect(od2d->met2DOD, SIGNAL(sendNetAPs(int, vector<pair<QString, float>>)), this, SLOT(setNetAPs(int, vector<pair<QString, float>>)));
		connect(od2d->met2DOD, SIGNAL(sendAvgIOUs(vector<float>, vector<float>)), this, SLOT(setAvgIOUs(vector<float>, vector<float>)));
		connect(ui->thSpinBox, SIGNAL(valueChanged(int)), od2d, SLOT(setThreshold(int)));
		connect(ui->accuracyButton, SIGNAL(clicked()), od2d, SLOT(calcAccuracy()));
		connect(this, SIGNAL(sendListIdx(int)), od2d, SLOT(setDataIdx(int)));
		connect(ui->leftButton, SIGNAL(clicked()), od2d, SLOT(goLeft()));
		connect(ui->rightButton, SIGNAL(clicked()), od2d, SLOT(goRight()));
		
	}else if(task_idx==1){
		//setting 2d semantic segmentation
		ui->net1Name->setText(" EfficientPS"); net1_achieve = 80.3;
		ui->net2Name->setText(" HRNet-OCR"); net2_achieve = 85.1;
		ss2d = new mode2DSS(this);
		ss2d->threshold = threshold;
		this->set2DSSLayouts();
		connect(ss2d, SIGNAL(sendImgList(QStringList)), this, SLOT(setImageList(QStringList)));
		connect(ss2d, SIGNAL(sendGTImg(QImage)), this, SLOT(setGTImage(QImage)));
		connect(ss2d, SIGNAL(sendNet1Img(QImage)), this, SLOT(setNet1Image(QImage)));
		connect(ss2d, SIGNAL(sendNet2Img(QImage)), this, SLOT(setNet2Image(QImage)));
		connect(ss2d, SIGNAL(sendStart()), this, SLOT(setLoadingMovie()));
		connect(ss2d, SIGNAL(sendStop()), this, SLOT(stopLoadingMovie()));
		connect(ss2d->met2DSS, SIGNAL(sendmIOUs(float, float)), this, SLOT(setmAPs(float, float)));
		connect(ss2d->met2DSS, SIGNAL(sendNetIOUs(int, vector<pair<QString, float>>)), this, SLOT(setNetAPs(int, vector<pair<QString, float>>)));
		connect(ss2d->met2DSS, SIGNAL(sendAvgIOU(float, float)), this, SLOT(setAvgIOU(float, float)));
		connect(ss2d->met2DSS, SIGNAL(sendAvgIOUs(vector<float>, vector<float>)), this, SLOT(setAvgIOUs(vector<float>, vector<float>)));
		connect(ui->thSpinBox, SIGNAL(valueChanged(int)), ss2d, SLOT(setThreshold(int)));
		connect(ui->accuracyButton, SIGNAL(clicked()), ss2d, SLOT(calcAccuracy()));
		connect(this, SIGNAL(sendListIdx(int)), ss2d, SLOT(setDataIdx(int)));
		connect(ui->leftButton, SIGNAL(clicked()), ss2d, SLOT(goLeft()));
		connect(ui->rightButton, SIGNAL(clicked()), ss2d, SLOT(goRight()));
	}else if(task_idx==2){
		//setting 3D Object Detection
		ui->net1Name->setText(" Voxel R-CNN"); net1_achieve = 81.6; //81.6
		ui->net2Name->setText(" PV R-CNN"); net2_achieve = 81.4; //81.4
		od3d = new mode3DOD(this);
		od3d->threshold = threshold;
		this->set3DODLayouts();
		connect(od3d, SIGNAL(sendPCDList(QStringList)), this, SLOT(setPCDList(QStringList)));
		connect(od3d, SIGNAL(sendAvgIOU(float, float)), this, SLOT(setAvgIOU(float, float)));
		connect(od3d, SIGNAL(sendGTPCD(QString, vector<BBoxes::BBox3D>)),this, SLOT(setGTPCD(QString, vector<BBoxes::BBox3D>)));
		connect(od3d->met3DOD, SIGNAL(sendmAPs(float, float)), this, SLOT(setmAPs(float, float)));
		connect(od3d->met3DOD, SIGNAL(sendNetAPs(int, vector<pair<QString, float>>)), this, SLOT(setNetAPs(int, vector<pair<QString, float>>)));
		connect(od3d->met3DOD, SIGNAL(sendAvgIOUs(vector<float>, vector<float>)), this, SLOT(setAvgIOUs(vector<float>, vector<float>)));
		connect(ui->thSpinBox, SIGNAL(valueChanged(int)), od3d, SLOT(setThreshold(int)));
		connect(ui->accuracyButton, SIGNAL(clicked()), od3d, SLOT(calcAccuracy()));
		connect(this, SIGNAL(sendListIdx(int)), od3d, SLOT(setDataIdx(int)));
		connect(ui->leftButton, SIGNAL(clicked()), od3d, SLOT(goLeft()));
		connect(ui->rightButton, SIGNAL(clicked()), od3d, SLOT(goRight()));
	}else{
		cout<<"There is no task to do "<<endl;
	}
}
void MainWindow::setServer(){
	QInputDialog qDialog;
	QStringList items;
	items << QString("192.168.0.1"); items << QString("192.168.0.2"); items << QString("192.168.0.3");

	qDialog.setOptions(QInputDialog::UseListViewForComboBoxItems);
	qDialog.setStyleSheet("QInputDialog {background-color: #F1F5F8;}");
	qDialog.setComboBoxItems(items);
	qDialog.setWindowTitle("Choose Server");
	if(qDialog.exec()) server = qDialog.textValue().toStdString();
}

void MainWindow::setData(){
	QString data_dir = QFileDialog::getExistingDirectory(this, "Select Top Driectory of Verification Dataum", QDir::currentPath(),QFileDialog::ShowDirsOnly);
    dataset_path = data_dir.toStdString();
	data_path =  QString::fromStdString(dataset_path+"/gt/data/"); 
	label_path = QString::fromStdString(dataset_path+"/gt/label/"); 
	
	if(task_idx==0){
		int n1_w = net1Label->width(); int n1_h =  net1Label->height();
		int gt_w = gtLabel->width(); int gt_h = gtLabel->height();
		od2d->n1_w = n1_w; od2d->n1_h = n1_h; 
		od2d->gt_w = gt_w; od2d->gt_h = gt_h;
		od2d->setData(dataset_path, data_path, label_path);
	}else if(task_idx==1){
		ss2d->setData(dataset_path);
	}else if(task_idx==2){
		od3d->setData(dataset_path, data_path, label_path);
	}else{
		cout<<"There is no task to do "<<endl;
	}
}

void MainWindow::set2DODLayouts(){
	gtLabel = new QLabel(this);
	net1Label = new QLabel(this);
	net2Label = new QLabel(this);
	ui->gtLayout->addWidget(gtLabel);
	ui->net1Layout1->addWidget(net1Label);
	ui->net2Layout1->addWidget(net2Label);
}

void MainWindow::set2DSSLayouts(){
	ui->net1AP_2->setText("Net1's mIOU");
	ui->net2AP_2->setText("Net2's mIOU");
	gtLabel = new QLabel(this);
	net1Label = new QLabel(this);
	net2Label = new QLabel(this);
	ui->gtLayout->addWidget(gtLabel);
	ui->net1Layout1->addWidget(net1Label);
	ui->net2Layout1->addWidget(net2Label);	
}

void MainWindow::set3DODLayouts(){
	
	vtkWg = new vtkWidget(this);
	vtkW1 = new vtkWidget(this);
	vtkW2 = new vtkWidget(this);

	connect(od3d, SIGNAL(sendGTPCD(QString, vector<BBoxes::BBox3D>)),vtkWg, SLOT(display_pcd(QString, vector<BBoxes::BBox3D>)));
	connect(od3d, SIGNAL(sendNet1PCD(QString, vector<BBoxes::BBox3D>)),vtkW1, SLOT(display_pcd(QString, vector<BBoxes::BBox3D>)));
	connect(od3d, SIGNAL(sendNet2PCD(QString, vector<BBoxes::BBox3D>)),vtkW2, SLOT(display_pcd(QString, vector<BBoxes::BBox3D>)));

	ui->gtLayout->layout()->addWidget(vtkWg);
	ui->net1Layout1->layout()->addWidget(vtkW1);
	ui->net2Layout1->layout()->addWidget(vtkW2);

	vtkWg->init();
	vtkW1->init();
	vtkW2->init();
}

void MainWindow::setLoadingMovie(){
	loadingLabel = new QLabel(this);
	loadingLabel->resize(100, 40); 
	loadingLabel->setStyleSheet("background-color: rgb(61, 67, 87);color: rgb(226, 230, 235);");
	loadingLabel->move(855,450);
	loadingLabel->setAlignment(Qt::AlignCenter);
	/*
	movie = new QMovie(":/gif/loader.gif");
	loadingLabel->setMovie(movie);
	movie->start();
	*/
	loadingLabel->setText("Processing . . .");
	loadingLabel->show();
}

void MainWindow::stopLoadingMovie(){
	//movie->stop();
	loadingLabel->hide();
}

void clearLayout(QLayout *layout){
	if(layout == NULL)
		return;
	while(QLayoutItem *item = layout->takeAt(0)){
		if(item->layout()){
			clearLayout(item->layout());
			delete item->layout();
		}
		if(item->widget()){
			delete item->widget();
		}
		delete item;
	}
}

void MainWindow::clearLayouts(){
	clearLayout(ui->gtLayout);
	clearLayout(ui->net1Layout1); clearLayout(ui->net1Layout2);
	clearLayout(ui->net2Layout1); clearLayout(ui->net2Layout2);
	clearLayout(ui->lodLayout);
	ui->dataList->clear();
	ui->net1AverageLabel->clear();
	ui->net2AverageLabel->clear();
	ui->net1AP->clear();
	ui->net2AP->clear();
	ui->levelLabel->clear();
}

void MainWindow::setListIdx(QListWidgetItem *qi){
	qi->isSelected();
	emit sendListIdx(ui->dataList->currentRow());
}

void MainWindow::setImageList(QStringList img_data_list){
	ui->informationLabel->setText(QString::number(img_data_list.size()));
	ui->dataList->addItems(img_data_list);
}

void MainWindow::setPCDList(QStringList pcd_data_list){
	ui->informationLabel->setText(QString::number(pcd_data_list.size()));
	ui->dataList->addItems(pcd_data_list);
}

void MainWindow::setGTImage(QImage _qimg){
	gtImg = _qimg;
	gtLabel->setPixmap(QPixmap::fromImage(_qimg).scaled(gtLabel->width(), gtLabel->height(), Qt::KeepAspectRatio));
	QCoreApplication::processEvents();
}

void MainWindow::setNet1Image(QImage _qimg){
	net1Label->setPixmap(QPixmap::fromImage(_qimg).scaled(net1Label->width(), net1Label->height(), Qt::KeepAspectRatio));
	QCoreApplication::processEvents();
}

void MainWindow::setNet2Image(QImage _qimg){
	net2Label->setPixmap(QPixmap::fromImage(_qimg).scaled(net2Label->width(), net2Label->height(), Qt::KeepAspectRatio));
	QCoreApplication::processEvents();
}

void MainWindow::setGTPCD(QString gtpath, vector<BBoxes::BBox3D> boxes){
	gtPCD_path = gtpath;
	gtPCD_boxes = boxes;
}

void MainWindow::setAvgIOU(float avg_net1_iou, float avg_net2_iou){
	avg_net1 = avg_net1_iou;
	avg_net2 = avg_net2_iou;
	this->setLOD();
}

void MainWindow::setLOD(){
	QString str1("Average IOU : "); QString str2("Average IOU : ");

	ui->net1AverageLabel->setText((str1.append(QString::number(avg_net1))).append("%"));
	ui->net2AverageLabel->setText((str2.append(QString::number(avg_net2))).append("%"));
	int LOD = this->getLOD(avg_net1, avg_net2);
	QPixmap lodPNG;
	if(LOD == 1){
		ui->levelLabel->setStyleSheet("QLabel{color: #3D4357; background-color: #58DE7E; border:3px; border-radius:50px}");
		ui->levelLabel->setText("Easy");
	}
	else if(LOD == 2){
		ui->levelLabel->setStyleSheet("QLabel{color: #3D4357; background-color: #FFC658; border:3px; border-radius:50px}");
		ui->levelLabel->setText("Moderate");
	}
	else if(LOD == 3){
		ui->levelLabel->setStyleSheet("QLabel{color: #3D4357; background-color: #FF7058; border:3px; border-radius:50px}");
		ui->levelLabel->setText("Hard");
	}
} 

void MainWindow::setAchieve(float AP1, float AP2){
	float achieve1, achieve2;
	if(AP1 <= 0.0) achieve1 = 0.0;
	else achieve1 = float(AP1/net1_achieve+1e-9)*100;
	if(AP2 <= 0.0) achieve2 = 0.0;
	else achieve2 = float(AP2/net2_achieve+1e-9)*100;
	if(achieve1 > 100) achieve1 = 100;
	if(achieve2 > 100) achieve2 = 100;
	ui->net1AP_4->setText(QString::number(achieve1).append("%"));
	ui->net2AP_4->setText(QString::number(achieve2).append("%"));
}

void MainWindow::setmAPs(float AP1, float AP2){
	ui->net1AP->setText(QString::number(AP1).append("%"));
	ui->net2AP->setText(QString::number(AP2).append("%"));
	this->setAchieve(AP1, AP2);
}

bool sortbysec(const pair<QString, float> & a, const pair<QString, float> &b){
	return (a.second > b.second);
}
void MainWindow::setNetAPs(int type, vector<pair<QString, float>> mAP){
	QBarSet *set0;
	if(task_idx == 1) set0 = new QBarSet("IOU(%)");
	else set0 = new QBarSet("AP(%)");
	sort(mAP.begin(), mAP.end(), sortbysec);
	if(mAP.size() > 10){
		for(size_t i=0; i<10; i++){
			if(mAP[i].second>0.0){
				*set0 << (mAP[i].second)*100;
			}
		}
	}else{
		for(size_t i=0; i<mAP.size(); i++){
			if(mAP[i].second>0.0){
				*set0 << (mAP[i].second)*100;
			}
		}
	}
	
	set0->setColor("#3D4357");
	QHorizontalBarSeries *series = new QHorizontalBarSeries();
	series->append(set0);
	series->setLabelsVisible(true);
	QChart *chart = new QChart();
	chart->addSeries(series);
	chart->setBackgroundVisible(false);
	if(type == 1){
		if(task_idx == 1)chart->setTitle("Network1's each Class's IOU(%)");
		else chart->setTitle("Network1's each Class's AP(%)");
	}else if(type == 2){
		if(task_idx == 1)chart->setTitle("Network2's each Class's IOU(%)");
		else chart->setTitle("Network2's each Class's AP(%)");
	}
	chart->setAnimationOptions(QChart::SeriesAnimations);
	QStringList categories;
	if(mAP.size()>10){
		for(size_t i=0; i<10; i++){
			if(mAP[i].second>0.0) categories << mAP[i].first;
		}
	}else{
		for(size_t i=0; i<mAP.size(); i++){
			if(mAP[i].second>0.0) categories << mAP[i].first;
		}
	}
	
	QBarCategoryAxis *axisY = new QBarCategoryAxis();
	axisY->append(categories);
	chart->addAxis(axisY, Qt::AlignLeft);
	series->attachAxis(axisY);
	QValueAxis *axisX = new QValueAxis();
	chart->addAxis(axisX, Qt::AlignBottom);
	series->attachAxis(axisX);
	axisX->setMin(0.0);
	axisX->setMax(100.0);
	chart->legend()->hide(); //setVisible(true);
	//chart->legend()->setAlignment(Qt::AlignBottom);
	QChartView *chartView = new QChartView(chart);
	chartView->setRenderHint(QPainter::Antialiasing);
	
	if(type == 1){
		while(QLayoutItem *wItem =ui->net1Layout2->takeAt(0))
			delete wItem;
		ui->net1Layout2->addWidget(chartView);
	}else if(type == 2){
		while(QLayoutItem *wItem =ui->net2Layout2->takeAt(0))
			delete wItem;
		ui->net2Layout2->addWidget(chartView);
	}
}

void MainWindow::setAvgIOUs(vector<float> net1, vector<float> net2){
	vector<int> lods;
	for(size_t i=0; i<net1.size(); i++){
		lods.push_back(this->getLOD(net1[i], net2[i]));
	}
	int lods_cnt[3] = {0,0,0};
	
	for(size_t i=0; i<lods.size(); i++){
		if(lods[i] == 1) 
			lods_cnt[0] += 1;
		else if(lods[i] == 2)
			lods_cnt[1] += 1;
		else if(lods[i] == 3)
			lods_cnt[2] += 1;
	}

	int max = lods_cnt[0] + lods_cnt[1] + lods_cnt[2];

	QBarSet *set0 = new QBarSet("E");
	QBarSet *set1 = new QBarSet("M");
	QBarSet *set2 = new QBarSet("H");

	*set0 << lods_cnt[0]; 
	*set1 << lods_cnt[1];
	*set2 << lods_cnt[2];

	set0->setColor("#58DE7E");
	set1->setColor("#FFC658");
	set2->setColor("#FF7058");

	QBarSeries *series = new QBarSeries();
	series->append(set0);
	series->append(set1);
	series->append(set2);
	series->setLabelsVisible(true);

	QChart *chart = new QChart();
	chart->addSeries(series);
	chart->setBackgroundVisible(false);
	chart->setTitle("Count of LOD");
	chart->setAnimationOptions(QChart::SeriesAnimations);
	/*
	QStringList categories;
	categories << "LoD";
	QBarCategoryAxis *axisX = new QBarCategoryAxis();
	axisX->append(categories);
	chart->addAxis(axisX, Qt::AlignBottom);
	series->attachAxis(axisX);
	*/
	QValueAxis *axisY = new QValueAxis();
	axisY->setRange(0, max);
	axisY->setLabelFormat("%d");
	chart->addAxis(axisY,Qt::AlignLeft);
	//series->attachAxis(axisY);
	chart->legend()->setVisible(true);
	chart->legend()->setAlignment(Qt::AlignBottom);
	QChartView *chartView = new QChartView(chart);
	chartView->setRenderHint(QPainter::Antialiasing);

	while(QLayoutItem *wItem =ui->lodLayout->takeAt(0))
		delete wItem;
	ui->lodLayout->addWidget(chartView);
}

void MainWindow::setThreshold(int _th){
	threshold = _th;
	this->setLOD();
}

int MainWindow::getLOD(float _net1, float _net2){
	/* Set target IOU threshold , defaul = 60 */
	float th = float(threshold);
	int LOD = 0;
	if( _net1 > th && _net2 > th) LOD = 1;
	else if(_net1 > th && _net2 < th) LOD = 2;
	else if(_net1 < th && _net2 > th) LOD = 2;
	else if(_net1 < th && _net2 < th) LOD = 3;
	else LOD = 0;
	return LOD;
}

void checkNmake(QString path){
	if(QDir(path).exists()){
		QDir(path).removeRecursively();
	}
	QDir().mkdir(path);
}

void MainWindow::setStorage(){
	QString storage_dir = QFileDialog::getExistingDirectory(this, "Select Directory to Storing Verified Data", QDir::currentPath(),QFileDialog::ShowDirsOnly);
    storage_path = storage_dir.toStdString()+ "/";

	checkNmake(QString::fromStdString(storage_path+"/accept"));
	checkNmake(QString::fromStdString(storage_path+"/reject"));
	checkNmake(QString::fromStdString(storage_path+"/accept/gt/"));
	checkNmake(QString::fromStdString(storage_path+"/accept/net1/"));
	checkNmake(QString::fromStdString(storage_path+"/accept/net2/"));
	checkNmake(QString::fromStdString(storage_path+"/reject/gt/"));
	checkNmake(QString::fromStdString(storage_path+"/reject/net1/"));
	checkNmake(QString::fromStdString(storage_path+"/reject/net2/"));
	checkNmake(QString::fromStdString(storage_path+"/accept/gt/data/"));
	checkNmake(QString::fromStdString(storage_path+"/reject/gt/data/"));
	
	if(task_idx == 0 || task_idx == 2){
		checkNmake(QString::fromStdString(storage_path+"/accept/gt/label/"));
		checkNmake(QString::fromStdString(storage_path+"/accept/net1/label/"));
		checkNmake(QString::fromStdString(storage_path+"/accept/net2/label/"));
		checkNmake(QString::fromStdString(storage_path+"/reject/gt/label/"));
		checkNmake(QString::fromStdString(storage_path+"/reject/net1/label"));
		checkNmake(QString::fromStdString(storage_path+"/reject/net2/label/"));
	}else if(task_idx == 1){
		checkNmake(QString::fromStdString(storage_path+"/accept/net1/data/"));
		checkNmake(QString::fromStdString(storage_path+"/accept/net2/data/"));
		checkNmake(QString::fromStdString(storage_path+"/reject/net1/data/"));
		checkNmake(QString::fromStdString(storage_path+"/reject/net2/data/"));
	}
}

void MainWindow::displayDetail(){
	if(task_idx == 0 || task_idx == 1){
		QWidget *dWindow = new QWidget;
		dWindow->resize(1280, 720);
		dWindow->setWindowTitle(QApplication::translate("displaywidget", "Display Widget"));
		dWindow->show();
		QLabel *dLabel = new QLabel(dWindow);
		dLabel->resize(1280, 720);
		dLabel->setPixmap(QPixmap::fromImage(gtImg).scaled(dLabel->width(), dLabel->height(), Qt::KeepAspectRatio));
		dLabel->show();
	}else if(task_idx==2){
		QWidget *dWindow = new QWidget;
		dWindow->resize(1280, 720);
		dWindow->setWindowTitle(QApplication::translate("displaywidget", "Display Widget"));
		dWindow->show();
		vtkWidget *dvtkW = new vtkWidget(dWindow);
		dvtkW->init();
		connect(this, SIGNAL(sendGTPCD(QString, vector<BBoxes::BBox3D>)), dvtkW, SLOT(display_pcd(QString, vector<BBoxes::BBox3D>)));
		emit sendGTPCD(gtPCD_path, gtPCD_boxes);
		dvtkW->resize(1280,720);
		dvtkW->show();
	}
}
void MainWindow::dataAccept(){
	if(task_idx == 0) od2d->saveAccept(storage_path);
	else if(task_idx == 1) ss2d->saveAccept(storage_path);
	else if(task_idx == 2) od3d->saveAccept(storage_path);
}
void MainWindow::dataReject(){
	if(task_idx == 0) od2d->saveReject(storage_path);
	else if(task_idx == 1) ss2d->saveReject(storage_path);
	else if(task_idx == 2) od3d->saveReject(storage_path);
}

