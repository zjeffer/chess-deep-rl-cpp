#pragma once

#include <QTextEdit>
#include <QScrollBar>

class Console : public QTextEdit {

  Q_OBJECT

  public:
	explicit Console(QWidget *parent = nullptr);

	void setLocalEchoEnabled(bool set);

  public slots:
	void putData(const QByteArray &text);

  signals:
	void getData(const QByteArray &text);

  protected:
	void keyPressEvent(QKeyEvent *e) override;
	void mousePressEvent(QMouseEvent *e) override;
	void mouseDoubleClickEvent(QMouseEvent *e) override;
	void contextMenuEvent(QContextMenuEvent *e) override;

  private:
	bool m_localEchoEnabled = false;

	QScrollBar* m_VerticalScrollBar;

	// queue of lines to be printed
	QStringList m_Lines;
};
