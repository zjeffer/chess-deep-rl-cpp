#include "console.hh"

Console::Console(QWidget *parent) : QPlainTextEdit(parent) {
	document()->setMaximumBlockCount(100);
	QPalette p = palette();
	p.setColor(QPalette::Base, Qt::black);
	p.setColor(QPalette::Text, Qt::green);
	setPalette(p);

	// don't wrap lines, use horizontal scrolling when lines are too long
	setLineWrapMode(QPlainTextEdit::LineWrapMode::NoWrap);

	// set font
	QFont font;
	font.setFamily("Monospace");
	font.setPointSize(10);
	setFont(font);

	// set scrollbar
	m_verticalScrollBar = verticalScrollBar();
}

void Console::putData(const QByteArray &text) {
	insertPlainText(text.toStdString().c_str());

	m_verticalScrollBar->setValue(m_verticalScrollBar->maximum());
}

void Console::setLocalEchoEnabled(bool set) { m_localEchoEnabled = set; }

void Console::keyPressEvent(QKeyEvent *e) {
	switch (e->key()) {
	case Qt::Key_Backspace:
	case Qt::Key_Left:
	case Qt::Key_Right:
	case Qt::Key_Up:
	case Qt::Key_Down:
		break;
	default:
		if (m_localEchoEnabled)
			QPlainTextEdit::keyPressEvent(e);
		emit getData(e->text().toLocal8Bit());
	}
}

void Console::mousePressEvent(QMouseEvent *e) {
	Q_UNUSED(e);
	setFocus();
}

void Console::mouseDoubleClickEvent(QMouseEvent *e) { Q_UNUSED(e); }

void Console::contextMenuEvent(QContextMenuEvent *e) { Q_UNUSED(e); }
