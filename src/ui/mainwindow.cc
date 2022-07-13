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
	std::cout << text << std::endl;
	this->m_console->putData(text.c_str());
}

MainWindow::~MainWindow() { 
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
	if (is_selfPlaying) {
		// stop selfplay
		ui->Button_Start->setText("Start self-play");
		this->print("Stopping selfplay");

		// TODO: stop selfplay thread
	} else {
		if (ui->Input_setDataFolder->text().length() == 0){
			this->print("Error: Data folder not set!");
			return;
		}
		if (ui->Input_setModel->text().length() == 0){
			this->print("Error: Model file not set!");
			return;
		}
		ui->Button_Start->setText("Stop self-play");
		this->print("Starting selfplay...");

		// start selfplay in new thread
		std::thread thread_selfplay = std::thread(
			&SelfPlay::playContinuously,
			ui->Input_setModel->text().toStdString(), // model path
			400, // amount of sims per move
			1 // amount of parallel games
		);
		thread_selfplay.detach();
	}

	is_selfPlaying = !is_selfPlaying;
}
















