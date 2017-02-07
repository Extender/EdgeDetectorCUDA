// Alexey Gavryushin / 2016 & 2017; this code is hereby released into the public domain.
// For suggestions, please contact int01@outlook.com

#include "mainwindow.h"
#include <QApplication>

#include "extcolordefs.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
