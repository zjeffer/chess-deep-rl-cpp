#pragma once

#include <QMainWindow>
#include <string>
#include "../neuralnet.hh"
#include "console.hh"


QT_BEGIN_NAMESPACE
namespace Ui {
	class MainWindow;
} // namespace Ui
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = nullptr);
	~MainWindow();

	void print(const std::string &text);

	void on_setModel_clicked();
	void on_setDataFolder_clicked();
	void on_startSelfPlay_clicked();

	void startSelfPlay();
	void stopSelfPlay();

	Console* getConsole() const;

private:
	Ui::MainWindow *m_Ui;
	Console* m_Console;

	std::vector<QMetaObject::Connection> m_Connections;

	std::shared_ptr<NeuralNetwork> m_NN;
};

