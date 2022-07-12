#pragma once

#include <QMainWindow>
#include <string>
#include "console.hh"

QT_BEGIN_NAMESPACE
namespace Ui {
    class MainWindow;
}
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

private:
	Ui::MainWindow *ui;

	Console* m_console;

	bool is_selfPlaying = false;

};
