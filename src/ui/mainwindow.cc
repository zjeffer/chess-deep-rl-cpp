#include <iostream>
#include <QFileDialog>
#include <qobjectdefs.h>
#include <QMessageBox>

#include "mainwindow.hh"
#include "./ui_mainwindow.h"
#include "../selfplay.hh"
#include "../common.hh"

MainWindow::MainWindow(QWidget *parent) : 
	QMainWindow(parent), 
	m_Ui(new Ui::MainWindow),
	m_Console(new Console)
	
	{
	m_Ui->setupUi(this);

	// init logger
	g_Logger = std::make_shared<Logger>(this);

	QVBoxLayout* mainBox = m_Ui->VBox_Main;
	mainBox->insertWidget(-1, m_Console);

	// set random seed
	g_Generator.seed(std::random_device{}());
	LOG(DEBUG) << "Test random value: " << g_Generator();

	// hold connections
	m_Connections = {
		// button actions
		connect(m_Ui->Button_setModel, &QPushButton::clicked, this, &MainWindow::on_setModel_clicked),
		connect(m_Ui->Button_setDataFolder, &QPushButton::clicked, this, &MainWindow::on_setDataFolder_clicked),
		connect(m_Ui->Button_Start, &QPushButton::clicked, this, &MainWindow::on_startSelfPlay_clicked),
		// menu actions
		connect(m_Ui->actionExit, &QAction::triggered, this, &MainWindow::close),
		// g3log <-> console connection
		connect(m_Console, &Console::getData, m_Console, &Console::putData)
	};

	// set input text
	m_Ui->Input_setDataFolder->setText(QDir::currentPath() + "/memory/");	
}

MainWindow::~MainWindow() { 
	std::cout << "MainWindow destructor" << std::endl;
	if (g_IsSelfPlaying){
		stopSelfPlay();
	}
	// disconnect signals
	for (auto& c : m_Connections) {
		disconnect(c);
	}
	g_Running = false;
	// delete m_Console;
	// delete m_Ui;
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


	LOG(INFO) << "Loaded model from: " + filename.toStdString();
	m_Ui->Input_setModel->setText(filename);
}

void MainWindow::on_setDataFolder_clicked() {
	// open file dialog to choose a folder
	QString dir = QFileDialog::getExistingDirectory(
		this, tr("Open data folder"), QDir::currentPath(), QFileDialog::ShowDirsOnly
	);

	if (dir.size() == 0) {
		LOG(INFO) << "Error: no folder selected";
		return;
	}

	LOG(INFO) << "Data folder set to " + dir.toStdString();
	m_Ui->Input_setDataFolder->setText(dir);
	// TODO: set data folder in options
}

void MainWindow::on_startSelfPlay_clicked(){
	if (g_IsSelfPlaying) {
		stopSelfPlay();
	} else {
		// check inputs
		if (m_Ui->Input_setDataFolder->text().length() == 0){
			LOG(INFO) << "Error: Data folder not set!";
			return;
		}
		if (m_Ui->Input_setModel->text().length() == 0){
			LOG(INFO) << "Error: Model file not set!";
			return;
		}
		startSelfPlay();
	}
}

void MainWindow::stopSelfPlay() {
	m_Ui->Button_Start->setText("Start self-play");
	g_IsSelfPlaying = false;

	LOG(INFO) << "Stopped selfplay.";
}

void MainWindow::startSelfPlay() {
	m_Ui->Button_Start->setText("Stop self-play");
	LOG(INFO) << "Starting selfplay with " + m_Ui->SpinBox_Threads->text().toStdString() + " threads, and " + m_Ui->SpinBox_Sims->text().toStdString() + " simulations/move...";

	// start selfplay
	g_IsSelfPlaying = true;

	// create the memory folder if it doesn't exist
	try {
		QDir dir(m_Ui->Input_setDataFolder->text());
		if (!dir.exists()) {
			dir.mkpath(".");
		}
	} catch (const std::exception& e) {
		LOG(FATAL) << "Error: Could not create memory folder!";
		exit(EXIT_FAILURE);
	}

	m_NN = std::make_shared<NeuralNetwork>(m_Ui->Input_setModel->text().toStdString(), m_Ui->RButton_CPU->isChecked());

	for (int t = 0; t < m_Ui->SpinBox_Threads->value(); t++) {
		std::thread thread_selfplay = std::thread(
			&SelfPlay::playContinuously,
			m_NN, // model
			m_Ui->SpinBox_Sims->value() // amount of sims per move
		);
		thread_selfplay.detach();
	}
}

Console* MainWindow::getConsole() const {
	return m_Console;
}