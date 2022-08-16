#pragma once

#include <QPlainTextEdit>

class Console : public QPlainTextEdit {

  Q_OBJECT

  signals:
	void getData(const QByteArray &text);

  public:
	explicit Console(QWidget *parent = nullptr);

	void putData(const QByteArray &text);
	void setLocalEchoEnabled(bool set);

  protected:
	void keyPressEvent(QKeyEvent *e) override;
	void mousePressEvent(QMouseEvent *e) override;
	void mouseDoubleClickEvent(QMouseEvent *e) override;
	void contextMenuEvent(QContextMenuEvent *e) override;

  private:
	bool m_localEchoEnabled = false;
};
