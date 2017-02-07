#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    MainWindowEx(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    scene=new GraphicsSceneEx();
    pixmapItem=new QGraphicsPixmapItem();
    scene->addItem(pixmapItem);
    ui->graphicsView->setScene(scene);
    connect(this,SIGNAL(windowResizedEx(QResizeEvent*)),this,SLOT(fitToWindow()));
    connect(ui->browseButton,SIGNAL(clicked(bool)),this,SLOT(browseButtonClicked()));
    connect(ui->cannyEdgeDetectionBtn,SIGNAL(clicked(bool)),this,SLOT(cannyBtnClicked()));
    connect(ui->robertsEdgeDetectionBtn,SIGNAL(clicked(bool)),this,SLOT(robertsBtnClicked()));
    connect(ui->sobelEdgeDetectionBtn,SIGNAL(clicked(bool)),this,SLOT(sobelBtnClicked()));
    connect(ui->gaussianBlurCUDABtn,SIGNAL(clicked(bool)),this,SLOT(gaussianBlurCUDABtnClicked()));
    connect(ui->cannyCUDAEdgeDetectionBtn,SIGNAL(clicked(bool)),this,SLOT(cannyCUDABtnClicked()));
    connect(ui->sobelCUDAEdgeDetectionBtn,SIGNAL(clicked(bool)),this,SLOT(sobelCUDABtnClicked()));
    connect(ui->robertsCUDAEdgeDetectionBtn,SIGNAL(clicked(bool)),this,SLOT(robertsCUDABtnClicked()));
    connect(ui->aboutBtn,SIGNAL(clicked(bool)),this,SLOT(aboutBtnClicked()));
    connect(ui->saveAsBtn,SIGNAL(clicked(bool)),this,SLOT(saveAsBtnClicked()));
    connect(ui->resetBtn,SIGNAL(clicked(bool)),this,SLOT(resetBtnClicked()));

    loadDialog=new QFileDialog(this);
    loadDialog->setNameFilter("All images (*.jpg *.jpeg *.png *.gif *.bmp *.dib *.tif *.tiff)");
    loadDialog->setDirectory(QApplication::applicationDirPath());
    connect(loadDialog,SIGNAL(fileSelected(QString)),this,SLOT(fileSelected(QString)));

    saveDialog=new QFileDialog(this);
    saveDialog->setAcceptMode(QFileDialog::AcceptSave);
    QStringList filters;
    filters<<"JPEG image (*.jpg)";
    filters<<"PNG image (*.png)";
    filters<<"Bitmap (*.bmp)";
    saveDialog->setNameFilters(filters);
    saveDialog->setDirectory(QApplication::applicationDirPath());
    connect(saveDialog,SIGNAL(fileSelected(QString)),this,SLOT(saveDialogFileSelected(QString)));

    originalImage=0;
    filteredImage=0;
    bmpData=0;
    filteredData=0;

    ui->gaussianDeviationBox->setValue(2.4);
    ui->cannyHighTresholdBox->setValue(0.25);
    ui->cannyLowTresholdBox->setValue(0.15);

    QString proposedFile=QApplication::applicationDirPath().replace("/","\\")+"\\example.bmp";
    QFile f(proposedFile);
    if(f.exists())
    {
        currentFile=proposedFile;
        ui->lineEdit->setText(currentFile);
        // Quick example image loading code:
        QTimer *tmr=new QTimer(this);
        connect(tmr,&QTimer::timeout,[=]() {
            fileSelected(ui->lineEdit->text());
            setWindowState(Qt::WindowMaximized);
        });
        tmr->setSingleShot(true);
        tmr->start(100);
    }
    else
        currentFile="";
}

MainWindow::~MainWindow()
{
    delete ui;
    delete originalImage;
    delete filteredImage;
    free(bmpData);
    free(filteredData);
}

QImage *MainWindow::getImageFromBmpData(int32_t width, int32_t height, uint32_t *data)
{
    QImage *image=new QImage(width,height,QImage::Format_ARGB32);
    image->fill(0xFFFFFFFF);
    for(int32_t y=0;y<height;y++)
    {
        for(int32_t x=0;x<width;x++)
        {
            image->setPixel(x,y,data[y*width+x]);
        }
    }
    return image;
}

float *MainWindow::applyGaussianBlurToSingleChannelFloatArray(float *in, int32_t width, int32_t height, int32_t filterSize, float deviation)
{
    int32_t filterSizeInPixels=2*filterSize+1;
    int32_t filterAreaInPixels=filterSizeInPixels*filterSizeInPixels;
    float *out=(float*)malloc(width*height*sizeof(float));

    float *filterFactors=(float*)malloc(filterAreaInPixels*sizeof(float));
    float preFactor=(1.0f/(2.0f*M_PI_F*deviation*deviation));
    float preFactor2=(2.0f*deviation*deviation);

    for(int32_t filterY=0;filterY<filterSizeInPixels;filterY++)
    {
        int32_t offset=filterY*filterSizeInPixels;
        for(int32_t filterX=0;filterX<filterSizeInPixels;filterX++)
        {
            int n1=filterX+1-filterSize-1;
            int n2=filterY+1-filterSize-1;
            float factor=preFactor*exp(-1.0f*((n1*n1+n2*n2)/preFactor2));
            filterFactors[offset+filterX]=factor;
        }
    }

    for(int32_t y=0;y<height;y++)
    {
        for(int32_t x=0;x<width;x++)
        {
            float pixelValueSum=0.0f;

            // Convolve using this pixel as the center

            for(int32_t yOfFilter=0;yOfFilter<filterSizeInPixels;yOfFilter++)
            {
                int32_t yWithFilter=-filterSize+y+yOfFilter;
                 // Use symmetry to compensate for missing pixels (in order to avoid dark borders)
                if(yWithFilter<0)
                    yWithFilter=0;
                else if(yWithFilter>=height)
                    yWithFilter=height-1;
                for(int32_t xOfFilter=0;xOfFilter<filterSizeInPixels;xOfFilter++)
                {
                    int32_t xWithFilter=-filterSize+x+xOfFilter;
                    // Use symmetry to compensate for missing pixels (in order to avoid dark borders)
                    if(xWithFilter<0)
                        xWithFilter=0;
                    else if(xWithFilter>=width)
                        xWithFilter=width-1;

                    float factor=filterFactors[yOfFilter*filterSizeInPixels+xOfFilter];

                    pixelValueSum+=in[yWithFilter*width+xWithFilter]*factor;
                }
            }

            out[y*width+x]=pixelValueSum;
        }
    }
    return out;
}

uint32_t *MainWindow::getImageFromBWFloatArray(float *in, int32_t width, int32_t height)
{
    uint32_t *out=(uint32_t*)malloc(width*height*sizeof(uint32_t));
    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            int32_t pos=offset+x;
            float pixelValue=in[pos];
            uint32_t component=(uint32_t)round(pixelValue*255);
            uint32_t outColor=0xFF000000;
            outColor|=(component<<16)|(component<<8)|component;
            out[pos]=outColor;
        }
    }
    return out;
}

uint32_t *MainWindow::qImageToBitmapData(QImage *image)
{
    int32_t width=image->width();
    int32_t height=image->height();
    uint32_t *out=(uint32_t*)malloc(width*height*sizeof(uint32_t));
    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        QRgb *scanLine=(QRgb*)image->scanLine(y); // Do not free!
        for(int32_t x=0;x<width;x++)
        {
            QRgb color=scanLine[x];
            uint32_t alpha=qAlpha(color);
            uint32_t red=qRed(color);
            uint32_t green=qGreen(color);
            uint32_t blue=qBlue(color);
            out[offset+x]=(alpha<<24)|(red<<16)|(green<<8)|blue;
        }
        // Do not free "scanLine"!
    }
    return out;
}

uint32_t *MainWindow::getImageFromChannelFloatArrays(float *rChannel, float *gChannel, float *bChannel, int32_t width, int32_t height)
{
    uint32_t *out=(uint32_t*)malloc(width*height*sizeof(uint32_t));
    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            int32_t pos=offset+x;
            uint32_t red=round(rChannel[pos]*255.0);
            uint32_t green=round(gChannel[pos]*255.0);
            uint32_t blue=round(bChannel[pos]*255.0);
            uint32_t outColor=0xFF000000;
            outColor|=(red<<16)|(green<<8)|blue;
            out[pos]=outColor;
        }
    }
    return out;
}

float *MainWindow::getBWFloatArrayFromImage(uint32_t *image, int32_t width, int32_t height)
{
    float *out=(float*)malloc(width*height*sizeof(float));
    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            uint32_t color=image[offset+x];
            out[offset+x]=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
        }
    }
    return out;
}

float *MainWindow::getRedChannelFloatArrayFromImage(uint32_t *image, int32_t width, int32_t height)
{
    float *out=(float*)malloc(width*height*sizeof(float));
    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            uint32_t color=image[offset+x];
            float value=((float)((color>>16)&0xff))/255.0;
            out[offset+x]=value;
        }
    }
    return out;
}

float *MainWindow::getGreenChannelFloatArrayFromImage(uint32_t *image, int32_t width, int32_t height)
{
    float *out=(float*)malloc(width*height*sizeof(float));
    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            uint32_t color=image[offset+x];
            float value=((float)((color>>8)&0xff))/255.0;
            out[offset+x]=value;
        }
    }
    return out;
}

float *MainWindow::getBlueChannelFloatArrayFromImage(uint32_t *image, int32_t width, int32_t height)
{
    float *out=(float*)malloc(width*height*sizeof(float));
    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            uint32_t color=image[offset+x];
            float value=((float)(color&0xff))/255.0;
            out[offset+x]=value;
        }
    }
    return out;
}

int32_t MainWindow::round(float in)
{
    int32_t f=floor(in);
    return in<0.0?((float)in-f>0.5?f+1:f):((float)in-f>=0.5?f+1:f);
}

void MainWindow::browseButtonClicked()
{
    loadDialog->exec();
}

void MainWindow::fileSelected(QString file)
{
    currentFile=file;
    ui->lineEdit->setText(file.replace("/","\\"));
    loadDialog->setDirectory(QFileInfo(file).absoluteDir());

    QFile f(currentFile);
    if(!f.exists())
    {
        QMessageBox::critical(this,"Error","The selected file does not exist.");
        return;
    }
    delete originalImage;
    originalImage=new QImage(currentFile);
    if(originalImage->isNull())
    {
        QMessageBox::critical(this,"Error","The format of the selected file is not supported.");
        originalImage=0;
        return;
    }
    free(bmpData);
    bmpData=qImageToBitmapData(originalImage);
    delete filteredImage;
    filteredImage=0;

    scene->setSceneRect(0,0,originalImage->width(),originalImage->height());
    pixmapItem->setPixmap(QPixmap::fromImage(*originalImage));
    ui->graphicsView->viewport()->update();
    fitToWindow();
}

void MainWindow::saveDialogFileSelected(QString file)
{
    if(originalImage==0||originalImage->isNull())
        return;

    QString format;
    int ind=file.lastIndexOf('.');
    if(ind==-1||ind==file.length()-1)
    {
        QMessageBox::critical(this,"Error","Could not save image.");
        return;
    }

    format=file.mid(ind+1);
    char *fStr=strdup(format.toStdString().c_str());
    if(!filteredImage->save(file,fStr,100))
         QMessageBox::critical(this,"Error","Could not save image.");
    free(fStr);
}

void MainWindow::cannyBtnClicked()
{
    if(originalImage==0||originalImage->isNull())
        return;

    int32_t width=originalImage->width();
    int32_t height=originalImage->height();

    float gaussianDeviation=ui->gaussianDeviationBox->value();
    int32_t gaussianFilterSize=floor(gaussianDeviation*3.0f); // NVidia standard
    float *bwImg=getBWFloatArrayFromImage(bmpData,width,height);
    float *gaussianImg=applyGaussianBlurToSingleChannelFloatArray(bwImg,width,height,gaussianFilterSize,gaussianDeviation);

    // Perform convolution (2 filters, accX and accY)

    float *gradient=(float*)malloc(width*height*sizeof(float));
    float *gradientAtan2=(float*)malloc(width*height*sizeof(float)); // In degs

    // Modify Sobel algorithm, too!

    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            float accXValueSum=0.0f;
            float accYValueSum=0.0f;

            int32_t leftX=x-1;
            int32_t rightX=x+1;
            int32_t topY=y-1;
            int32_t bottomY=y+1;

            bool hasTop=topY>=0;
            bool hasBottom=bottomY<height;
            bool hasLeft=leftX>=0;
            bool hasRight=rightX<width;
            float factor;

            if(hasLeft)
            {
                if(hasTop)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=gaussianImg[topY*width+leftX]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=gaussianImg[topY*width+leftX]*factor;
                }
                else
                {
                    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
                    // avoid false positives on the borders of the image

                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=gaussianImg[y*width+leftX]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=gaussianImg[y*width+leftX]*factor;
                }
                if(hasBottom)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=gaussianImg[bottomY*width+leftX]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=gaussianImg[bottomY*width+leftX]*factor;
                }
                else
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=gaussianImg[y*width+leftX]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=gaussianImg[y*width+leftX]*factor;
                }
                // X accumulator
                factor=-2.0f;
                accXValueSum+=gaussianImg[y*width+leftX]*factor;
            }
            else
            {
                if(hasTop)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=gaussianImg[topY*width+x]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=gaussianImg[topY*width+x]*factor;
                }
                else
                {
                    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
                    // avoid false positives on the borders of the image

                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=gaussianImg[y*width+x]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=gaussianImg[y*width+x]*factor;
                }
                if(hasBottom)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=gaussianImg[bottomY*width+x]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=gaussianImg[bottomY*width+x]*factor;
                }
                else
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=gaussianImg[y*width+x]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=gaussianImg[y*width+x]*factor;
                }
                // X accumulator
                factor=-2.0f;
                accXValueSum+=gaussianImg[y*width+x]*factor;
            }
            if(hasRight)
            {
                if(hasTop)
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=gaussianImg[topY*width+rightX]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=gaussianImg[topY*width+rightX]*factor;
                }
                else
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=gaussianImg[y*width+rightX]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=gaussianImg[y*width+rightX]*factor;
                }
                if(hasBottom)
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=gaussianImg[bottomY*width+rightX]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=gaussianImg[bottomY*width+rightX]*factor;
                }
                else
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=gaussianImg[y*width+rightX]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=gaussianImg[y*width+rightX]*factor;
                }
                // X accumulator
                factor=2.0f;
                accXValueSum+=gaussianImg[y*width+rightX]*factor;
            }
            else
            {
                if(hasTop)
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=gaussianImg[topY*width+x]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=gaussianImg[topY*width+x]*factor;
                }
                else
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=gaussianImg[y*width+x]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=gaussianImg[y*width+x]*factor;
                }
                if(hasBottom)
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=gaussianImg[bottomY*width+x]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=gaussianImg[bottomY*width+x]*factor;
                }
                else
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=gaussianImg[y*width+x]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=gaussianImg[y*width+x]*factor;
                }
                // X accumulator
                factor=2.0f;
                accXValueSum+=gaussianImg[y*width+x]*factor;
            }
            if(hasTop)
            {
                // Y accumulator
                factor=-2.0f;
                accYValueSum+=gaussianImg[topY*width+x]*factor;
            }
            else
            {
                // Y accumulator
                factor=-2.0f;
                accYValueSum+=gaussianImg[y*width+x]*factor;
            }
            if(hasBottom)
            {
                // Y accumulator
                factor=2.0f;
                accYValueSum+=gaussianImg[bottomY*width+x]*factor;
            }
            else
            {
                // Y accumulator
                factor=2.0f;
                accYValueSum+=gaussianImg[y*width+x]*factor;
            }
            // The pixel in the center has 0.0f as its factor for both filters.

            int32_t pos=offset+x;

            // Result
            gradient[pos]=sqrt(accXValueSum*accXValueSum+accYValueSum*accYValueSum);
            gradientAtan2[pos]=atan2(accXValueSum,accYValueSum);
        }
    }

    float *preResult=(float*)malloc(width*height*sizeof(float));

    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            int32_t leftX=x-1;
            int32_t rightX=x+1;
            int32_t topY=y-1;
            int32_t bottomY=y+1;

            bool hasTop=topY>=0;
            bool hasBottom=bottomY<height;
            bool hasLeft=leftX>=0;
            bool hasRight=rightX<width;

            int32_t pos=offset+x;
            float angle=gradientAtan2[pos];

            int rAngle=(int)round(angle/(0.25f*M_PI_F)); // 45 degrees
            if(rAngle<0)
                rAngle=4+rAngle;
            bool eastWest=rAngle==0||rAngle==4; // The first section is split (one half on each end)
            bool northEastSouthWest=rAngle==1;
            bool northSouth=rAngle==2;
            bool northWestSouthEast=rAngle==3;

            float thisPixelValue=gradient[y*width+x];
            if(eastWest)
            {
                float neighborPixelValue1=hasTop?gradient[topY*width+x]:gradient[y*width+x];
                float neighborPixelValue2=hasBottom?gradient[bottomY*width+x]:gradient[y*width+x];
                if(thisPixelValue>neighborPixelValue1&&thisPixelValue>neighborPixelValue2)
                    preResult[pos]=thisPixelValue;
                else
                    preResult[pos]=0.0f;
            }
            else if(northEastSouthWest)
            {
                float neighborPixelValue1=hasLeft&&hasTop?gradient[topY*width+leftX]:(hasLeft?gradient[y*width+leftX]:(hasTop?gradient[topY*width+x]:gradient[y*width+x]));
                float neighborPixelValue2=hasRight&&hasBottom?gradient[bottomY*width+rightX]:(hasRight?gradient[y*width+rightX]:(hasBottom?gradient[bottomY*width+x]:gradient[y*width+x]));
                if(thisPixelValue>neighborPixelValue1&&thisPixelValue>neighborPixelValue2)
                    preResult[pos]=thisPixelValue;
                else
                    preResult[pos]=0.0f;
            }
            else if(northSouth)
            {
                float neighborPixelValue1=hasLeft?gradient[y*width+leftX]:gradient[y*width+x];
                float neighborPixelValue2=hasRight?gradient[y*width+rightX]:gradient[y*width+x];
                if(thisPixelValue>neighborPixelValue1&&thisPixelValue>neighborPixelValue2)
                    preResult[pos]=thisPixelValue;
                else
                    preResult[pos]=0.0f;
            }
            else if(northWestSouthEast)
            {
                float neighborPixelValue1=hasRight&&hasTop?gradient[topY*width+rightX]:(hasRight?gradient[y*width+rightX]:(hasTop?gradient[topY*width+x]:gradient[y*width+x]));
                float neighborPixelValue2=hasLeft&&hasBottom?gradient[bottomY*width+leftX]:(hasLeft?gradient[y*width+leftX]:(hasBottom?gradient[bottomY*width+x]:gradient[y*width+x]));
                if(thisPixelValue>neighborPixelValue1&&thisPixelValue>neighborPixelValue2)
                    preResult[pos]=thisPixelValue;
                else
                    preResult[pos]=0.0f;
            }
        }
    }

    float lowTreshold=ui->cannyLowTresholdBox->value();
    float highTreshold=ui->cannyHighTresholdBox->value();

    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            float value=preResult[offset+x];
            float newValue;
            if(value<lowTreshold)
                newValue=0.0f;
            else if(value<highTreshold)
                newValue=0.5f;
            else // if(value>=highTreshold)
                newValue=1.0f;
            preResult[offset+x]=newValue;
        }
    }

    float *result=(float*)malloc(width*height*sizeof(float));

    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            float value=preResult[offset+x];
            if(value==0.0f) // No edge; do not keep
                continue;
            else if(value==1.0f) // Strong edge; keep
                goto Keep;

            // Weak edge; decide whether to keep it (if at least one neighboring pixel is a strong edge)

            {
                int32_t leftX=x-1;
                int32_t rightX=x+1;
                int32_t topY=y-1;
                int32_t bottomY=y+1;

                bool hasTop=topY>=0;
                bool hasBottom=bottomY<height;
                bool hasLeft=leftX>=0;
                bool hasRight=rightX<width;
                // Use x==1.0f, not x>0.0f here!
                if(hasLeft)
                {
                    if(preResult[y*width+leftX]==1.0f)
                        goto Keep;
                    if(hasTop)
                    {
                        if(preResult[topY*width+leftX]==1.0f)
                            goto Keep;
                    }
                    if(hasBottom)
                    {
                        if(preResult[bottomY*width+leftX]==1.0f)
                            goto Keep;
                    }
                }
                if(hasRight)
                {
                    if(preResult[y*width+rightX]==1.0f)
                        goto Keep;
                    if(hasTop)
                    {
                        if(preResult[topY*width+rightX]==1.0f)
                            goto Keep;
                    }
                    if(hasBottom)
                    {
                        if(preResult[bottomY*width+rightX]==1.0f)
                            goto Keep;
                    }
                }
                if(hasTop)
                {
                    if(preResult[topY*width+x]==1.0f)
                        goto Keep;
                }
                if(hasBottom)
                {
                    if(preResult[bottomY*width+x]==1.0f)
                        goto Keep;
                }
                // Do not keep
                result[offset+x]=0.0f;
                continue;
            }
            Keep:
            // Keep
            result[offset+x]=1.0f;
        }
    }

    free(filteredData);
    filteredData=getImageFromBWFloatArray(result,width,height);
    delete filteredImage;
    filteredImage=new QImage((uchar*)filteredData,width,height,QImage::Format_ARGB32);
    pixmapItem->setPixmap(QPixmap::fromImage(*filteredImage));

    // Free buffers

    free(bwImg);
    free(gaussianImg);
    free(preResult);
    free(result);
    free(gradient);
    free(gradientAtan2);
}

void MainWindow::robertsBtnClicked()
{
    if(originalImage==0||originalImage->isNull())
        return;

    int32_t width=originalImage->width();
    int32_t height=originalImage->height();

    // Perform convolution (2 filters, accX and accY)

    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
    // avoid false positives on the borders of the image

    float *bwImg=getBWFloatArrayFromImage(bmpData,width,height);
    float *result=(float*)malloc(width*height*sizeof(float));
    float amplifier=ui->amplifierBox->value();

    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            float accXValueSum=0.0f;
            float accYValueSum=0.0f;

            int32_t rightX=x+1;
            int32_t bottomY=y+1;

            bool hasBottom=bottomY<height;
            bool hasRight=rightX<width;
            float factor;

            if(hasBottom)
            {
                if(hasRight)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[bottomY*width+rightX]*factor;
                }
                else
                {
                    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
                    // avoid false positives on the borders of the image

                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[bottomY*width+x]*factor;
                }
                // Y accumulator
                factor=-1.0f;
                accYValueSum+=bwImg[bottomY*width+x]*factor;
            }
            else
            {
                if(hasRight)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[y*width+rightX]*factor;
                }
                else
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[y*width+x]*factor;
                }
                // Y accumulator
                factor=-1.0f;
                accYValueSum+=bwImg[y*width+x]*factor;
            }
            if(hasRight)
            {
                // Y accumulator
                factor=1.0f;
                accYValueSum+=bwImg[y*width+rightX]*factor;
            }
            else
            {
                // Y accumulator
                factor=1.0f;
                accYValueSum+=bwImg[y*width+x]*factor;
            }
            // This pixel:
            // X accumulator
            factor=1.0f;
            accXValueSum+=bwImg[y*width+x]*factor;

            int32_t pos=offset+x;

            // Result
            result[pos]=amplifier*sqrt(accXValueSum*accXValueSum+accYValueSum*accYValueSum);
        }
    }
    free(filteredData);
    filteredData=getImageFromBWFloatArray(result,width,height);
    delete filteredImage;
    filteredImage=new QImage((uchar*)filteredData,width,height,QImage::Format_ARGB32);
    pixmapItem->setPixmap(QPixmap::fromImage(*filteredImage));
    free(bwImg);
    free(result);
}

void MainWindow::sobelBtnClicked()
{
    // Modify Canny edge detection algorithm, too!

    if(originalImage==0||originalImage->isNull())
        return;

    int32_t width=originalImage->width();
    int32_t height=originalImage->height();

    // Perform convolution (2 filters, accX and accY)

    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
    // avoid false positives on the borders of the image

    float *bwImg=getBWFloatArrayFromImage(bmpData,width,height);
    float *result=(float*)malloc(width*height*sizeof(float));
    float amplifier=ui->amplifierBox->value();

    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        for(int32_t x=0;x<width;x++)
        {
            float accXValueSum=0.0f;
            float accYValueSum=0.0f;

            int32_t leftX=x-1;
            int32_t rightX=x+1;
            int32_t topY=y-1;
            int32_t bottomY=y+1;

            bool hasTop=topY>=0;
            bool hasBottom=bottomY<height;
            bool hasLeft=leftX>=0;
            bool hasRight=rightX<width;
            float factor;

            if(hasLeft)
            {
                if(hasTop)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[topY*width+leftX]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=bwImg[topY*width+leftX]*factor;
                }
                else
                {
                    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
                    // avoid false positives on the borders of the image

                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[y*width+leftX]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=bwImg[y*width+leftX]*factor;
                }
                if(hasBottom)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[bottomY*width+leftX]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=bwImg[bottomY*width+leftX]*factor;
                }
                else
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[y*width+leftX]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=bwImg[y*width+leftX]*factor;
                }
                // X accumulator
                factor=-2.0f;
                accXValueSum+=bwImg[y*width+leftX]*factor;
            }
            else
            {
                if(hasTop)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[topY*width+x]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=bwImg[topY*width+x]*factor;
                }
                else
                {
                    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
                    // avoid false positives on the borders of the image

                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[y*width+x]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=bwImg[y*width+x]*factor;
                }
                if(hasBottom)
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[bottomY*width+x]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=bwImg[bottomY*width+x]*factor;
                }
                else
                {
                    // X accumulator
                    factor=-1.0f;
                    accXValueSum+=bwImg[y*width+x]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=bwImg[y*width+x]*factor;
                }
                // X accumulator
                factor=-2.0f;
                accXValueSum+=bwImg[y*width+x]*factor;
            }
            if(hasRight)
            {
                if(hasTop)
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=bwImg[topY*width+rightX]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=bwImg[topY*width+rightX]*factor;
                }
                else
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=bwImg[y*width+rightX]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=bwImg[y*width+rightX]*factor;
                }
                if(hasBottom)
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=bwImg[bottomY*width+rightX]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=bwImg[bottomY*width+rightX]*factor;
                }
                else
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=bwImg[y*width+rightX]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=bwImg[y*width+rightX]*factor;
                }
                // X accumulator
                factor=2.0f;
                accXValueSum+=bwImg[y*width+rightX]*factor;
            }
            else
            {
                if(hasTop)
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=bwImg[topY*width+x]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=bwImg[topY*width+x]*factor;
                }
                else
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=bwImg[y*width+x]*factor;
                    // Y accumulator
                    factor=-1.0f;
                    accYValueSum+=bwImg[y*width+x]*factor;
                }
                if(hasBottom)
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=bwImg[bottomY*width+x]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=bwImg[bottomY*width+x]*factor;
                }
                else
                {
                    // X accumulator
                    factor=1.0f;
                    accXValueSum+=bwImg[y*width+x]*factor;
                    // Y accumulator
                    factor=1.0f;
                    accYValueSum+=bwImg[y*width+x]*factor;
                }
                // X accumulator
                factor=2.0f;
                accXValueSum+=bwImg[y*width+x]*factor;
            }
            if(hasTop)
            {
                // Y accumulator
                factor=-2.0f;
                accYValueSum+=bwImg[topY*width+x]*factor;
            }
            else
            {
                // Y accumulator
                factor=-2.0f;
                accYValueSum+=bwImg[y*width+x]*factor;
            }
            if(hasBottom)
            {
                // Y accumulator
                factor=2.0f;
                accYValueSum+=bwImg[bottomY*width+x]*factor;
            }
            else
            {
                // Y accumulator
                factor=2.0f;
                accYValueSum+=bwImg[y*width+x]*factor;
            }
            // The pixel in the center has 0.0f as its factor for both filters.

            int32_t pos=offset+x;

            // Result
            result[pos]=amplifier*sqrt(accXValueSum*accXValueSum+accYValueSum*accYValueSum);
        }
    }
    free(filteredData);
    filteredData=getImageFromBWFloatArray(result,width,height);
    delete filteredImage;
    filteredImage=new QImage((uchar*)filteredData,width,height,QImage::Format_ARGB32);
    pixmapItem->setPixmap(QPixmap::fromImage(*filteredImage));
    free(bwImg);
    free(result);
}

void MainWindow::gaussianBlurCUDABtnClicked()
{
    if(originalImage==0||originalImage->isNull())
        return;

    if(originalImage->hasAlphaChannel())
    {
        int btn=QMessageBox::warning(this,"Warning","Alpha channel will be removed from image. Continue?",QMessageBox::Yes,QMessageBox::No);
        if(btn==QMessageBox::No)
            return;
    }

    int32_t width=originalImage->width();
    int32_t height=originalImage->height();
    float gaussianDeviation=ui->gaussianDeviationBox->value();
    int32_t gaussianFilterSize=floor(gaussianDeviation*3.0); // NVidia standard

    uint32_t *newImageData=cuda3ChannelGaussianBlur(bmpData,width,height,gaussianFilterSize,gaussianDeviation);
    if(newImageData==0)
    {
        QMessageBox::critical(this,"Error","Image too large to process or filter size too large!");
        return;
    }
    free(filteredData);
    filteredData=newImageData;
    delete filteredImage;
    filteredImage=new QImage((uchar*)filteredData,width,height,QImage::Format_ARGB32);
    pixmapItem->setPixmap(QPixmap::fromImage(*filteredImage));
}

void MainWindow::sobelCUDABtnClicked()
{
    if(originalImage==0||originalImage->isNull())
        return;

    int32_t width=originalImage->width();
    int32_t height=originalImage->height();
    float amplifier=ui->amplifierBox->value();

    uint32_t *newImageData=cudaSobelEdgeDetect(bmpData,width,height,amplifier);
    if(newImageData==0)
    {
        QMessageBox::critical(this,"Error","Image too large to process!");
        return;
    }
    free(filteredData);
    filteredData=newImageData;
    delete filteredImage;
    filteredImage=new QImage((uchar*)filteredData,width,height,QImage::Format_ARGB32);
    pixmapItem->setPixmap(QPixmap::fromImage(*filteredImage));
}

void MainWindow::robertsCUDABtnClicked()
{
    if(originalImage==0||originalImage->isNull())
        return;

    int32_t width=originalImage->width();
    int32_t height=originalImage->height();
    float amplifier=ui->amplifierBox->value();

    uint32_t *newImageData=cudaRobertsEdgeDetect(bmpData,width,height,amplifier);
    if(newImageData==0)
    {
        QMessageBox::critical(this,"Error","Image too large to process!");
        return;
    }
    free(filteredData);
    filteredData=newImageData;
    delete filteredImage;
    filteredImage=new QImage((uchar*)filteredData,width,height,QImage::Format_ARGB32);
    pixmapItem->setPixmap(QPixmap::fromImage(*filteredImage));
}

void MainWindow::cannyCUDABtnClicked()
{
    if(originalImage==0||originalImage->isNull())
        return;

    int32_t width=originalImage->width();
    int32_t height=originalImage->height();
    float deviation=ui->gaussianDeviationBox->value();
    float highTreshold=ui->cannyHighTresholdBox->value();
    float lowTreshold=ui->cannyLowTresholdBox->value();

    uint32_t *newImageData=cudaCannyEdgeDetect(bmpData,width,height,deviation,highTreshold,lowTreshold);
    if(newImageData==0)
    {
        QMessageBox::critical(this,"Error","Image too large to process!");
        return;
    }
    free(filteredData);
    filteredData=newImageData;
    delete filteredImage;
    filteredImage=new QImage((uchar*)filteredData,width,height,QImage::Format_ARGB32);
    pixmapItem->setPixmap(QPixmap::fromImage(*filteredImage));
}

void MainWindow::saveAsBtnClicked()
{
    if(originalImage==0||originalImage->isNull())
        return;

    saveDialog->exec();
}

void MainWindow::resetBtnClicked()
{
    if(originalImage==0||originalImage->isNull())
        return;

    pixmapItem->setPixmap(QPixmap::fromImage(*originalImage));
    delete filteredImage;
    filteredImage=0;
}

void MainWindow::aboutBtnClicked()
{
    QMessageBox::information(this,"About this program","This is a working demonstration of the Roberts cross, Sobel and Canny edge detection algorithms. All algorithms are implemented both as a CPU and a GPU (NVidia only) version. Because the GPU versions make use of parallelism, they are usually much faster than their CPU counterparts.\nTry loading an image and clicking one of the buttons!\n\nVisit http://github.com/Extender/EdgeDetectorCUDA/ for more information.\n\nAlexey Gavryushin / 2016 & 2017");
}

void MainWindow::fitToWindow()
{
    if(originalImage==0||originalImage->isNull())
        return;
    int width=originalImage->width();
    int height=originalImage->height();
    QRect rect=ui->graphicsView->contentsRect();
    int availableWidth=rect.width()-ui->graphicsView->verticalScrollBar()->width();
    int availableHeight=rect.height()-ui->graphicsView->horizontalScrollBar()->height();
    if((width-availableWidth)>(height-availableHeight))
        ui->graphicsView->setZoomFactor((float)((float)availableWidth)/((float)width));
    else if(height>availableHeight)
        ui->graphicsView->setZoomFactor((float)((float)availableHeight)/((float)height));
    else
        ui->graphicsView->setZoomFactor(1.0);
}

void MainWindow::resetZoom()
{
    ui->graphicsView->setZoomFactor(1.0);
}
