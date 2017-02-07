#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <stdint.h>
#include <math.h>

#include <QMainWindow>
#include <QFile>
#include <QImage>
#include <QFileDialog>
#include <QMessageBox>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QTimer>

#include "mainwindowex.h"
#include "graphicsviewex.h"
#include "graphicssceneex.h"

#include "extcolordefs.h"
#include "decimal.h"

// BEGIN FORWARD-DECLARATIONS OF CUDA FUNCTIONS

uint32_t *cudaGetBWImage(uint32_t *imageData,int width,int height);
uint32_t *cudaCannyEdgeDetect(uint32_t *imageData,int width,int height,float deviation,float highTreshold,float lowTreshold);
uint32_t *cudaSobelEdgeDetect(uint32_t *imageData,int width,int height,float amplifier);
uint32_t *cudaRobertsEdgeDetect(uint32_t *imageData,int width,int height,float amplifier);
uint32_t *cuda3ChannelGaussianBlur(uint32_t *imageData,int width,int height,int filterSize,float deviation);

// END FORWARD-DECLARATIONS OF CUDA FUNCTIONS

namespace Ui {
class MainWindow;
}

class MainWindow : public MainWindowEx
{
    Q_OBJECT

    QFileDialog *loadDialog;
    QFileDialog *saveDialog;
    QString currentFile;
    QImage *originalImage;
    QImage *filteredImage;
    GraphicsSceneEx *scene;
    QGraphicsPixmapItem *pixmapItem;
    uint32_t *bmpData;
    uint32_t *filteredData;

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    QImage *getImageFromBmpData(int32_t width,int32_t height,uint32_t *data);
    static float *applyGaussianBlurToSingleChannelFloatArray(float *in,int32_t width,int32_t height,int32_t filterSize,float deviation);
    static uint32_t *getImageFromBWFloatArray(float *in,int32_t width,int32_t height);
    static uint32_t *qImageToBitmapData(QImage *image);
    static uint32_t *getImageFromChannelFloatArrays(float *rChannel,float *gChannel,float *bChannel,int32_t width,int32_t height);
    static float *getBWFloatArrayFromImage(uint32_t *image,int32_t width,int32_t height);
    static float *getRedChannelFloatArrayFromImage(uint32_t *image,int32_t width,int32_t height);
    static float *getGreenChannelFloatArrayFromImage(uint32_t *image,int32_t width,int32_t height);
    static float *getBlueChannelFloatArrayFromImage(uint32_t *image,int32_t width,int32_t height);
    static int32_t round(float in);

public slots:
    void browseButtonClicked();
    void fileSelected(QString file);
    void saveDialogFileSelected(QString file);
    void cannyBtnClicked();
    void robertsBtnClicked();
    void sobelBtnClicked();
    void gaussianBlurCUDABtnClicked();
    void sobelCUDABtnClicked();
    void robertsCUDABtnClicked();
    void cannyCUDABtnClicked();
    void saveAsBtnClicked();
    void resetBtnClicked();
    void aboutBtnClicked();
    void fitToWindow();
    void resetZoom();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
