#include "console.hh"

Console::Console(QWidget *parent) : QTextEdit(parent) {
	document()->setMaximumBlockCount(100);
	QPalette p = palette();
	p.setColor(QPalette::Base, Qt::black);
	p.setColor(QPalette::Text, Qt::green);
	setPalette(p);

	// don't wrap lines, use horizontal scrolling when lines are too long
	setLineWrapMode(QTextEdit::LineWrapMode::NoWrap);

	// set font
	QFont font;
	font.setFamily("Monospace");
	font.setPointSize(11);
	setFont(font);

	// set scrollbar
	m_VerticalScrollBar = verticalScrollBar();
}

void Console::putData(const QByteArray &text) {
	insertPlainText(text.toStdString().c_str());

	if (!m_VerticalScrollBar->isSliderDown()) {
		// scroll to bottom if not dragging the scrollbar
		m_VerticalScrollBar->setValue(m_VerticalScrollBar->maximum());
	}
}

void Console::setLocalEchoEnabled(bool set) { m_localEchoEnabled = set; }

void Console::keyPressEvent(QKeyEvent *e) { Q_UNUSED(e); }

void Console::mousePressEvent(QMouseEvent *e) {
	Q_UNUSED(e);
	setFocus();
}

void Console::mouseDoubleClickEvent(QMouseEvent *e) { Q_UNUSED(e); }

void Console::contextMenuEvent(QContextMenuEvent *e) { Q_UNUSED(e); }
