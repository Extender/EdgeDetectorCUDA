#include "mainwindowex.h"

MainWindowEx::MainWindowEx(QWidget *parent) : QMainWindow(parent)
{

}

void MainWindowEx::resizeEvent(QResizeEvent *event)
{
    windowResizedEx(event);
    QMainWindow::resizeEvent(event);
}
