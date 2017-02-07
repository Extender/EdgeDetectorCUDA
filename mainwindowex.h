#ifndef MAINWINDOWEX_H
#define MAINWINDOWEX_H

#include <QMainWindow>
#include <QResizeEvent>

class MainWindowEx : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindowEx(QWidget *parent = 0);
    void resizeEvent(QResizeEvent *event);

signals:
    void windowResizedEx(QResizeEvent *event);

public slots:
};

#endif // MAINWINDOWEX_H
