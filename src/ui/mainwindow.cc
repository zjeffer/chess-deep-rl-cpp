#include <iostream>
#include <QFileDialog>

#include "mainwindow.hh"
#include "./ui_mainwindow.h"
#include "../selfplay.hh"
#include "../common.hh"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow) {
	ui->setupUi(this);

	QVBoxLayout* mainBox = ui->VBox_Main;
	m_console = new Console(this);
	mainBox->insertWidget(-1, m_console);

	this->print("Loaded console");


	// set button actions
	connect(ui->Button_setModel, &QPushButton::clicked, this, &MainWindow::on_setModel_clicked);
	connect(ui->Button_setDataFolder, &QPushButton::clicked, this, &MainWindow::on_setDataFolder_clicked);
	connect(ui->Button_Start, &QPushButton::clicked, this, &MainWindow::on_startSelfPlay_clicked);

	// set input text
	ui->Input_setDataFolder->setText(QDir::currentPath() + "/data/");

	connect(ui->actionExit, &QAction::triggered, this, &MainWindow::close);

}

void MainWindow::print(const std::string &text) {
	LOG(INFO) << text;
	this->m_console->putData(text.c_str());
}

MainWindow::~MainWindow() { 
	if (g_isSelfPlaying){
		stopSelfPlay();
	}
	g_running = false;
	delete ui;
}


void MainWindow::on_setModel_clicked() {
	// open file dialog
	QString filename = QFileDialog::getOpenFileName(
		nullptr,
		"Open model file",
		QDir::currentPath(),
		"Model files (*.pt)"
	);

	if (filename.size() == 0){
		return;
	}


	this->print("Loaded model from: " + filename.toStdString());
	ui->Input_setModel->setText(filename);
	// TODO: load model from filename
}

void MainWindow::on_setDataFolder_clicked() {
	// open file dialog to choose a folder
	QString dir = QFileDialog::getExistingDirectory(
		this, tr("Open data folder"), QDir::currentPath(), QFileDialog::ShowDirsOnly
	);

	if (dir.size() == 0) {
		this->print("Error: no folder selected");
		return;
	}

	this->print("Data folder set to " + dir.toStdString());
	ui->Input_setDataFolder->setText(dir);
	// TODO: set data folder in options
}

void MainWindow::on_startSelfPlay_clicked(){
	// TODO: start & stop selfplay
	if (g_isSelfPlaying) {
		stopSelfPlay();
	} else {
		// check inputs
		if (ui->Input_setDataFolder->text().length() == 0){
			this->print("Error: Data folder not set!");
			return;
		}
		if (ui->Input_setModel->text().length() == 0){
			this->print("Error: Model file not set!");
			return;
		}
		startSelfPlay();
	}
}

void MainWindow::stopSelfPlay() {
	ui->Button_Start->setText("Start self-play");
	g_isSelfPlaying = false;

	this->print("Stopped selfplay.");
	// delete nn;
}

void MainWindow::startSelfPlay() {
	ui->Button_Start->setText("Stop self-play");
	this->print("Starting selfplay with " + ui->SpinBox_Threads->text().toStdString() + " threads, and " + ui->SpinBox_Sims->text().toStdString() + "simulations/move...");

	// start selfplay
	g_isSelfPlaying = true;

	nn = new NeuralNetwork(ui->Input_setModel->text().toStdString(), ui->RButton_CPU->isChecked());

	for (int t = 0; t < ui->SpinBox_Threads->value(); t++) {
		std::thread thread_selfplay = std::thread(
			&SelfPlay::playContinuously,
			nn, // model
			400, // amount of sims per move
			1 // amount of parallel games
		);
		thread_selfplay.detach();
	}
	
	
}




