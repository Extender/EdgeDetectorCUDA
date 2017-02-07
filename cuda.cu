#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "decimal.h"
#include "extcolordefs.h"
#include "text.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <iostream>
#include <math.h>

// Optimal values (calculated)

#define CUDA_BLOCK_SIDE_LENGTH 32
#define CUDA_GAUSSIAN_BLUR_BLOCK_SIDE_LENGTH 16
#define CUDA_THREADS_PER_BLOCK (CUDA_BLOCK_SIDE_LENGTH*CUDA_BLOCK_SIDE_LENGTH)
#define CUDA_3_CHANNEL_MAX_FILTER_SIZE 19
#define CUDA_1_CHANNEL_MAX_FILTER_SIZE 19
#define CUDA_CANNY_LOW_COLOR 0xff000000
#define CUDA_CANNY_HIGH_COLOR 0xffffffff

#if 0==1
#define __shared__
#define __global__
#define __device__
#define __host__
#endif

void cudaLog(const char *str)
{
    std::cout<<str<<std::endl;
}

__global__ void device_getBWImage(uint32_t *imageData,int width,uint32_t *bwImageDataOut)
{
    int threadId=threadIdx.x;
    int y=blockIdx.y;
    int blockIdInLine=blockIdx.x;
    int x=blockIdInLine*blockDim.x+threadId;

    if(x>=width)
        return;

    size_t pos=y*width+x;
    uint32_t color=imageData[pos];
    float r=getFRed(color);
    float g=getFGreen(color);
    float b=getFBlue(color);

    uint8_t component=round((0.2126f*r+0.7152f*g+0.0722f*b)*255.0f);

    bwImageDataOut[pos]=component;
}

__global__ void device_cannyEdgeDetect_stage1(uint32_t *imageData,int width,int height,int filterSize,float deviation,float *gaussianBwArrayOut)
{
    const int blockImageSectionSingleChannelSizeSqrt=CUDA_1_CHANNEL_MAX_FILTER_SIZE*2+CUDA_BLOCK_SIDE_LENGTH;
    __shared__ float blockImageSection[blockImageSectionSingleChannelSizeSqrt*blockImageSectionSingleChannelSizeSqrt];

    int filterSizeInPixels=2*filterSize+1;
    int blockHorId=blockIdx.x;
    int blockVerId=blockIdx.y;
    int relativeX=threadIdx.x;
    int x=blockHorId*blockDim.x+relativeX;
    int relativeY=threadIdx.y;
    int y=blockVerId*blockDim.y+relativeY;

    int rowSize=2*filterSize+blockDim.x;
    int columnSize=2*filterSize+blockDim.y;
    int topOffsetRows=filterSize;

    if(x>=width||y>=height)
        return;

    size_t pos=y*width+x;
    uint32_t color=imageData[pos];
    // Use getFBWComponentFromColor
    float fbwComponent=getFBWComponentFromColor(color);

    // Copy pixels

    // Copy this pixel from source image

    int posInBlockImageSectionData=(topOffsetRows+relativeY)*rowSize+filterSize+relativeX;
    blockImageSection[posInBlockImageSectionData]=fbwComponent;

    // Copy extra pixels, if needed

    int effectiveBlockWidth=blockHorId==gridDim.x-1?(width-blockHorId*blockDim.x):blockDim.x;
    int effectiveBlockHeight=blockVerId==gridDim.y-1?(height-blockVerId*blockDim.y):blockDim.y;

    int relativeLeftExtra=-filterSize+relativeX;
    int relativeRightExtra=relativeX+filterSize;
    int relativeTopExtra=-filterSize+relativeY;
    int relativeBottomExtra=relativeY+filterSize;

    bool copyExtraFromLeft=false;
    bool leftOverflow=false;
    bool copyExtraFromRight=false;
    bool rightOverflow=false;
    int absLeftExtra;
    int absRightExtra;
    uint32_t extraColor;

    if(copyExtraFromLeft=(relativeLeftExtra<0))
    {
        absLeftExtra=x-filterSize;
        if(leftOverflow=(absLeftExtra<0))
        {
            extraColor=imageData[y*width];
            blockImageSection[posInBlockImageSectionData-filterSize]=getFBWComponentFromColor(extraColor);
        }
        else
        {
            extraColor=imageData[y*width+absLeftExtra];
            blockImageSection[posInBlockImageSectionData-filterSize]=getFBWComponentFromColor(extraColor);
        }
    }
    if(copyExtraFromRight=(relativeRightExtra>=effectiveBlockWidth))
    {
        absRightExtra=x+filterSize;
        if(rightOverflow=(absRightExtra>=width))
        {
            extraColor=imageData[y*width+width-1];
            blockImageSection[posInBlockImageSectionData+filterSize]=getFBWComponentFromColor(extraColor);
        }
        else
        {
            extraColor=imageData[y*width+absRightExtra];
            blockImageSection[posInBlockImageSectionData+filterSize]=getFBWComponentFromColor(extraColor);
        }
    }
    if(relativeTopExtra<0)
    {
        int absTopExtra=y-filterSize;
        if(absTopExtra<0)
        {
            extraColor=imageData[(-absTopExtra)*width+x];
            blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)]=getFBWComponentFromColor(extraColor);
            if(copyExtraFromLeft)
            {
                if(leftOverflow)
                {
                    extraColor=imageData[(-absTopExtra)*width];
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)-filterSize]=getFBWComponentFromColor(extraColor);
                }
                else
                {
                    extraColor=imageData[(-absTopExtra)*width+absLeftExtra];
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)-filterSize]=getFBWComponentFromColor(extraColor);
                }
            }
            if(copyExtraFromRight)
            {
                if(rightOverflow)
                {
                    extraColor=imageData[(-absTopExtra)*width+width-1];
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)+filterSize]=getFBWComponentFromColor(extraColor);
                }
                else
                {
                    extraColor=imageData[(-absTopExtra)*width+absRightExtra];
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)+filterSize]=getFBWComponentFromColor(extraColor);
                }
            }
        }
        else
        {
            extraColor=imageData[absTopExtra*width+x];
            blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)]=getFBWComponentFromColor(extraColor);
            if(copyExtraFromLeft)
            {
                if(leftOverflow)
                {
                    extraColor=imageData[absTopExtra*width];
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)-filterSize]=getFBWComponentFromColor(extraColor);
                }
                else
                {
                    extraColor=imageData[absTopExtra*width+absLeftExtra];
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)-filterSize]=getFBWComponentFromColor(extraColor);
                }
            }
            if(copyExtraFromRight)
            {
                if(rightOverflow)
                {
                    extraColor=imageData[absTopExtra*width+width-1];
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)+filterSize]=getFBWComponentFromColor(extraColor);
                }
                else
                {
                    extraColor=imageData[absTopExtra*width+absRightExtra];
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)+filterSize]=getFBWComponentFromColor(extraColor);
                }
            }
        }
    }
    if(relativeBottomExtra>=effectiveBlockHeight)
    {
        int absBottomExtra=y+filterSize;
        if(absBottomExtra>=height)
        {
            extraColor=imageData[(height-(absBottomExtra-height+1))*width+x];
            blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)]=getFBWComponentFromColor(extraColor);
            if(copyExtraFromLeft)
            {
                if(leftOverflow)
                {
                    extraColor=imageData[(height-(absBottomExtra-height+1))*width];
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)-filterSize]=getFBWComponentFromColor(extraColor);
                }
                else
                {
                    extraColor=imageData[(height-(absBottomExtra-height+1))*width+absLeftExtra];
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)-filterSize]=getFBWComponentFromColor(extraColor);
                }
            }
            if(copyExtraFromRight)
            {
                if(rightOverflow)
                {
                    extraColor=imageData[(height-(absBottomExtra-height+1))*width+width-1];
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)+filterSize]=getFBWComponentFromColor(extraColor);
                }
                else
                {
                    extraColor=imageData[(height-(absBottomExtra-height+1))*width+absRightExtra];
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)+filterSize]=getFBWComponentFromColor(extraColor);
                }
            }
        }
        else
        {
            extraColor=imageData[absBottomExtra*width+x];
            blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)]=getFBWComponentFromColor(extraColor);
            if(copyExtraFromLeft)
            {
                if(leftOverflow)
                {
                    extraColor=imageData[absBottomExtra*width];
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)-filterSize]=getFBWComponentFromColor(extraColor);
                }
                else
                {
                    extraColor=imageData[absBottomExtra*width+absLeftExtra];
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)-filterSize]=getFBWComponentFromColor(extraColor);
                }
            }
            if(copyExtraFromRight)
            {
                if(rightOverflow)
                {
                    extraColor=imageData[absBottomExtra*width+width-1];
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)+filterSize]=getFBWComponentFromColor(extraColor);
                }
                else
                {
                    extraColor=imageData[absBottomExtra*width+absRightExtra];
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)+filterSize]=getFBWComponentFromColor(extraColor);
                }
            }
        }
    }

    float preFactor=(1.0f/(2.0f*M_PI_F*deviation*deviation));
    float preFactor2=(2.0f*deviation*deviation);
    float filterFactors[CUDA_3_CHANNEL_MAX_FILTER_SIZE*CUDA_3_CHANNEL_MAX_FILTER_SIZE];
    for(int filterY=0;filterY<filterSizeInPixels;filterY++)
    {
        int offset=filterY*filterSizeInPixels;
        for(int filterX=0;filterX<filterSizeInPixels;filterX++)
        {
            int n1=filterX+1-filterSize-1;
            int n2=filterY+1-filterSize-1;
            float factor=preFactor*exp(-(((float)(n1*n1+n2*n2))/preFactor2));
            filterFactors[offset+filterX]=factor;
        }
    }

    __syncthreads();

    int relativeXWithBorderPixels=relativeX+filterSize;
    int relativeYWithBorderPixels=relativeY+filterSize;
    float pixelValueSum=0.0f;

    for(int yOfFilter=0;yOfFilter<filterSizeInPixels;yOfFilter++)
    {
        int yWithFilter=-filterSize+relativeYWithBorderPixels+yOfFilter;
         // Use symmetry to compensate for missing pixels (in order to avoid dark borders)
        if(yWithFilter<0)
            yWithFilter=(relativeYWithBorderPixels+filterSize)-yOfFilter;
        else if(yWithFilter>=columnSize)
            yWithFilter=(relativeYWithBorderPixels-filterSize)+(filterSizeInPixels-1-yOfFilter);
        for(int xOfFilter=0;xOfFilter<filterSizeInPixels;xOfFilter++)
        {
            int xWithFilter=-filterSize+relativeXWithBorderPixels+xOfFilter;
            // Use symmetry to compensate for missing pixels (in order to avoid dark borders)
            if(xWithFilter<0)
                xWithFilter=(relativeXWithBorderPixels+filterSize)-xOfFilter;
            else if(xWithFilter>=rowSize)
                xWithFilter=(relativeXWithBorderPixels-filterSize)+(filterSizeInPixels-1-xOfFilter);

            float factor=filterFactors[yOfFilter*filterSizeInPixels+xOfFilter];
            pixelValueSum+=blockImageSection[yWithFilter*rowSize+xWithFilter]*factor;
        }
    }

    gaussianBwArrayOut[pos]=pixelValueSum;
}

__global__ void device_cannyEdgeDetect_stage2(int width,int height,int filterSize,float *gaussianBwArray,float *gradientArrayOut,float *gradientAtan2ArrayOut)
{
    __shared__ float blockBWCache[(CUDA_BLOCK_SIDE_LENGTH+2)*(CUDA_BLOCK_SIDE_LENGTH+2)];

    int blockHorId=blockIdx.x;
    int blockVerId=blockIdx.y;
    int relativeX=threadIdx.x;
    int x=blockHorId*blockDim.x+relativeX;
    int relativeY=threadIdx.y;
    int y=blockVerId*blockDim.y+relativeY;

    if(x>=width||y>=height)
        return;

    size_t pos=y*width+x;

    float bw=gaussianBwArray[pos];

    int blockBWCacheRowSize=blockDim.x+2;

    int blockBWCachePos=(1+relativeY)*blockBWCacheRowSize+1+relativeX;
    blockBWCache[blockBWCachePos]=bw;

    bool hasTop=(y-1)>=0;
    bool hasBottom=(y+1)<height;
    bool hasLeft=(x-1)>=0;
    bool hasRight=(x+1)<width;

    bool leftmostInBlock=relativeX==0;
    bool rightmostInBlock=relativeX==blockDim.x-1||!(hasRight); // Blocks on the borders of the image have other border pixels
    bool topmostInBlock=relativeY==0;
    bool bottommostInBlock=relativeY==blockDim.y-1||!(hasBottom); // Blocks on the borders of the image have other border pixels

    if(leftmostInBlock&&hasLeft)
    {
        blockBWCache[blockBWCachePos-1]=gaussianBwArray[pos-1];
    }
    else if(rightmostInBlock&&hasRight)
    {
        blockBWCache[blockBWCachePos+1]=gaussianBwArray[pos+1];
    }

    if(topmostInBlock&&hasTop)
    {
        blockBWCache[blockBWCachePos-blockBWCacheRowSize]=gaussianBwArray[pos-width];
        if(leftmostInBlock&&hasLeft)
        {
            blockBWCache[blockBWCachePos-blockBWCacheRowSize-1]=gaussianBwArray[pos-width-1];
        }
        else if(rightmostInBlock&&hasRight)
        {
            blockBWCache[blockBWCachePos-blockBWCacheRowSize+1]=gaussianBwArray[pos-width+1];
        }
    }
    else if(bottommostInBlock&&hasBottom)
    {
        blockBWCache[blockBWCachePos+blockBWCacheRowSize]=gaussianBwArray[pos+width];
        if(leftmostInBlock&&hasLeft)
        {
            blockBWCache[blockBWCachePos+blockBWCacheRowSize-1]=gaussianBwArray[pos+width-1];
        }
        else if(rightmostInBlock&&hasRight)
        {
            blockBWCache[blockBWCachePos+blockBWCacheRowSize+1]=gaussianBwArray[pos+width+1];
        }
    }

    float accXValueSum=0.0f;
    float accYValueSum=0.0f;

    int relativeLeftX=relativeX-1;
    int relativeRightX=relativeX+1;
    int relativeYWithOffset=(1+relativeY)*blockBWCacheRowSize+1;
    int relativeTopYWithOffset=(1+relativeY-1)*blockBWCacheRowSize+1;
    int relativeBottomYWithOffset=(1+relativeY+1)*blockBWCacheRowSize+1;

    __syncthreads(); // Make sure blockBWCache is filled out correctly (see above)

    if(hasLeft)
    {
        if(hasTop)
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeTopYWithOffset+relativeLeftX]*(-1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeTopYWithOffset+relativeLeftX]*(-1.0f);
        }
        else
        {
            // Extend the image by 1 pixel on each side and using the outmost pixels in order to
            // avoid false positives on the borders of the image

            // X accumulator
            accXValueSum+=blockBWCache[relativeYWithOffset+relativeLeftX]*(-1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeYWithOffset+relativeLeftX]*(-1.0f);
        }
        if(hasBottom)
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeBottomYWithOffset+relativeLeftX]*(-1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeBottomYWithOffset+relativeLeftX]*(1.0f);
        }
        else
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeYWithOffset+relativeLeftX]*(-1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeYWithOffset+relativeLeftX]*(1.0f);
        }
        // X accumulator
        accXValueSum+=blockBWCache[relativeYWithOffset+relativeLeftX]*(-2.0f);
    }
    else
    {
        if(hasTop)
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeTopYWithOffset+relativeX]*(-1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeTopYWithOffset+relativeX]*(-1.0f);
        }
        else
        {
            // Extend the image by 1 pixel on each side and using the outmost pixels in order to
            // avoid false positives on the borders of the image

            // X accumulator
            accXValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(-1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(-1.0f);
        }
        if(hasBottom)
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeBottomYWithOffset+relativeX]*(-1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeBottomYWithOffset+relativeX]*(1.0f);
        }
        else
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(-1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(1.0f);
        }
        // X accumulator
        accXValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(-2.0f);
    }
    if(hasRight)
    {
        if(hasTop)
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeTopYWithOffset+relativeRightX]*(1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeTopYWithOffset+relativeRightX]*(-1.0f);
        }
        else
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeYWithOffset+relativeRightX]*(1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeYWithOffset+relativeRightX]*(-1.0f);
        }
        if(hasBottom)
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeBottomYWithOffset+relativeRightX]*(1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeBottomYWithOffset+relativeRightX]*(1.0f);
        }
        else
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeYWithOffset+relativeRightX]*(1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeYWithOffset+relativeRightX]*(1.0f);
        }
        // X accumulator
        accXValueSum+=blockBWCache[relativeYWithOffset+relativeRightX]*(2.0f);
    }
    else
    {
        if(hasTop)
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeTopYWithOffset+relativeX]*(1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeTopYWithOffset+relativeX]*(-1.0f);
        }
        else
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(-1.0f);
        }
        if(hasBottom)
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeBottomYWithOffset+relativeX]*(1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeBottomYWithOffset+relativeX]*(1.0f);
        }
        else
        {
            // X accumulator
            accXValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(1.0f);
            // Y accumulator
            accYValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(1.0f);
        }
        // X accumulator
        accXValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(2.0f);
    }
    if(hasTop)
    {
        // Y accumulator
        accYValueSum+=blockBWCache[relativeTopYWithOffset+relativeX]*(-2.0f);
    }
    else
    {
        // Y accumulator
        accYValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(-2.0f);
    }
    if(hasBottom)
    {
        // Y accumulator
        accYValueSum+=blockBWCache[relativeBottomYWithOffset+relativeX]*(2.0f);
    }
    else
    {
        // Y accumulator
        accYValueSum+=blockBWCache[relativeYWithOffset+relativeX]*(2.0f);
    }
    // The pixel in the center has 0.0f as its factor for both filters.

    // Result
    float gradient=sqrt(accXValueSum*accXValueSum+accYValueSum*accYValueSum);
    float gradientAtan2=atan2(accXValueSum,accYValueSum);

    gradientArrayOut[pos]=gradient;
    gradientAtan2ArrayOut[pos]=gradientAtan2;
}

__global__ void device_cannyEdgeDetect_stage3(int width,int height,float highTreshold,float lowTreshold,float *gradientArray,float *gradientAtan2Array,uint8_t *preResultArrayOut)
{
    __shared__ float blockGradientCache[(CUDA_BLOCK_SIDE_LENGTH+2)*(CUDA_BLOCK_SIDE_LENGTH+2)];

    int blockHorId=blockIdx.x;
    int blockVerId=blockIdx.y;
    int relativeX=threadIdx.x;
    int x=blockHorId*blockDim.x+relativeX;
    int relativeY=threadIdx.y;
    int y=blockVerId*blockDim.y+relativeY;

    if(x>=width||y>=height)
        return;

    size_t pos=y*width+x;

    float gradient=gradientArray[pos];

    int blockGradientCacheRowSize=blockDim.x+2;

    int blockGradientCachePos=(1+relativeY)*blockGradientCacheRowSize+1+relativeX;
    blockGradientCache[blockGradientCachePos]=gradient;

    bool hasTop=(y-1)>=0;
    bool hasBottom=(y+1)<height;
    bool hasLeft=(x-1)>=0;
    bool hasRight=(x+1)<width;

    bool leftmostInBlock=relativeX==0;
    bool rightmostInBlock=relativeX==blockDim.x-1||!(hasRight); // Blocks on the borders of the image have other border pixels
    bool topmostInBlock=relativeY==0;
    bool bottommostInBlock=relativeY==blockDim.y-1||!(hasBottom); // Blocks on the borders of the image have other border pixels

    if(leftmostInBlock&&hasLeft)
    {
        blockGradientCache[blockGradientCachePos-1]=gradientArray[pos-1];
    }
    else if(rightmostInBlock&&hasRight)
    {
        blockGradientCache[blockGradientCachePos+1]=gradientArray[pos+1];
    }

    if(topmostInBlock&&hasTop)
    {
        blockGradientCache[blockGradientCachePos-blockGradientCacheRowSize]=gradientArray[pos-width];
        if(leftmostInBlock&&hasLeft)
        {
            blockGradientCache[blockGradientCachePos-blockGradientCacheRowSize-1]=gradientArray[pos-width-1];
        }
        else if(rightmostInBlock&&hasRight)
        {
            blockGradientCache[blockGradientCachePos-blockGradientCacheRowSize+1]=gradientArray[pos-width+1];
        }
    }
    else if(bottommostInBlock&&hasBottom)
    {
        blockGradientCache[blockGradientCachePos+blockGradientCacheRowSize]=gradientArray[pos+width];
        if(leftmostInBlock&&hasLeft)
        {
            blockGradientCache[blockGradientCachePos+blockGradientCacheRowSize-1]=gradientArray[pos+width-1];
        }
        else if(rightmostInBlock&&hasRight)
        {
            blockGradientCache[blockGradientCachePos+blockGradientCacheRowSize+1]=gradientArray[pos+width+1];
        }
    }

    __syncthreads();

    float angle=gradientAtan2Array[pos];

    int rAngle=(int)round(angle/(0.25f*M_PI_F)); // 45 degrees
    if(rAngle<0)
        rAngle=4+rAngle;
    bool eastWest=rAngle==0||rAngle==4; // The first section is split (one half on each end)
    bool northEastSouthWest=rAngle==1;
    bool northSouth=rAngle==2;
    bool northWestSouthEast=rAngle==3;

    float preResult;

    if(eastWest)
    {
        float neighborPixelValue1=hasTop?blockGradientCache[blockGradientCachePos-blockGradientCacheRowSize]:gradient;
        float neighborPixelValue2=hasBottom?blockGradientCache[blockGradientCachePos+blockGradientCacheRowSize]:gradient;
        if(gradient>neighborPixelValue1&&gradient>neighborPixelValue2)
            preResult=gradient;
        else
            preResult=0.0f;
    }
    else if(northEastSouthWest)
    {
        float neighborPixelValue1=hasLeft&&hasTop?blockGradientCache[blockGradientCachePos-blockGradientCacheRowSize-1]:(hasLeft?blockGradientCache[blockGradientCachePos-1]:(hasTop?blockGradientCache[blockGradientCachePos-blockGradientCacheRowSize]:gradient));
        float neighborPixelValue2=hasRight&&hasBottom?blockGradientCache[blockGradientCachePos+blockGradientCacheRowSize+1]:(hasRight?blockGradientCache[blockGradientCachePos+1]:(hasBottom?blockGradientCache[blockGradientCachePos+blockGradientCacheRowSize]:gradient));
        if(gradient>neighborPixelValue1&&gradient>neighborPixelValue2)
            preResult=gradient;
        else
            preResult=0.0f;
    }
    else if(northSouth)
    {
        float neighborPixelValue1=hasLeft?blockGradientCache[blockGradientCachePos-1]:gradient;
        float neighborPixelValue2=hasRight?blockGradientCache[blockGradientCachePos+1]:gradient;
        if(gradient>neighborPixelValue1&&gradient>neighborPixelValue2)
            preResult=gradient;
        else
            preResult=0.0f;
    }
    else if(northWestSouthEast)
    {
        float neighborPixelValue1=hasRight&&hasTop?blockGradientCache[blockGradientCachePos-blockGradientCacheRowSize+1]:(hasRight?blockGradientCache[blockGradientCachePos+1]:(hasTop?blockGradientCache[blockGradientCachePos-blockGradientCacheRowSize]:gradient));
        float neighborPixelValue2=hasLeft&&hasBottom?blockGradientCache[blockGradientCachePos+blockGradientCacheRowSize-1]:(hasLeft?blockGradientCache[blockGradientCachePos-1]:(hasBottom?blockGradientCache[blockGradientCachePos+blockGradientCacheRowSize]:gradient));
        if(gradient>neighborPixelValue1&&gradient>neighborPixelValue2)
            preResult=gradient;
        else
            preResult=0.0f;
    }

    uint8_t preResultOut;

    if(preResult<lowTreshold)
        preResultOut=0;
    else if(preResult<highTreshold)
        preResultOut=1;
    else // if(preResult>=highTreshold)
        preResultOut=2;

    preResultArrayOut[pos]=preResultOut;
}

__global__ void device_cannyEdgeDetect_stage4(int width,int height,uint8_t *preResultArray,uint32_t *imageDataOut)
{
    __shared__ uint8_t blockPreResultCache[(CUDA_BLOCK_SIDE_LENGTH+2)*(CUDA_BLOCK_SIDE_LENGTH+2)];

    int blockHorId=blockIdx.x;
    int blockVerId=blockIdx.y;
    int relativeX=threadIdx.x;
    int x=blockHorId*blockDim.x+relativeX;
    int relativeY=threadIdx.y;
    int y=blockVerId*blockDim.y+relativeY;

    if(x>=width||y>=height)
        return;

    size_t pos=y*width+x;

    uint8_t preResult=preResultArray[pos];

    int blockPreResultCacheRowSize=blockDim.x+2;

    int blockPreResultCachePos=(1+relativeY)*blockPreResultCacheRowSize+1+relativeX;
    blockPreResultCache[blockPreResultCachePos]=preResult;

    bool hasTop=(y-1)>=0;
    bool hasBottom=(y+1)<height;
    bool hasLeft=(x-1)>=0;
    bool hasRight=(x+1)<width;

    bool leftmostInBlock=relativeX==0;
    bool rightmostInBlock=relativeX==blockDim.x-1||!(hasRight); // Blocks on the borders of the image have other border pixels
    bool topmostInBlock=relativeY==0;
    bool bottommostInBlock=relativeY==blockDim.y-1||!(hasBottom); // Blocks on the borders of the image have other border pixels

    if(leftmostInBlock&&hasLeft)
    {
        blockPreResultCache[blockPreResultCachePos-1]=preResultArray[pos-1];
    }
    else if(rightmostInBlock&&hasRight)
    {
        blockPreResultCache[blockPreResultCachePos+1]=preResultArray[pos+1];
    }

    if(topmostInBlock&&hasTop)
    {
        blockPreResultCache[blockPreResultCachePos-blockPreResultCacheRowSize]=preResultArray[pos-width];
        if(leftmostInBlock&&hasLeft)
        {
            blockPreResultCache[blockPreResultCachePos-blockPreResultCacheRowSize-1]=preResultArray[pos-width-1];
        }
        else if(rightmostInBlock&&hasRight)
        {
            blockPreResultCache[blockPreResultCachePos-blockPreResultCacheRowSize+1]=preResultArray[pos-width+1];
        }
    }
    else if(bottommostInBlock&&hasBottom)
    {
        blockPreResultCache[blockPreResultCachePos+blockPreResultCacheRowSize]=preResultArray[pos+width];
        if(leftmostInBlock&&hasLeft)
        {
            blockPreResultCache[blockPreResultCachePos+blockPreResultCacheRowSize-1]=preResultArray[pos+width-1];
        }
        else if(rightmostInBlock&&hasRight)
        {
            blockPreResultCache[blockPreResultCachePos+blockPreResultCacheRowSize+1]=preResultArray[pos+width+1];
        }
    }

    __syncthreads();

    if(preResult==0)
        imageDataOut[pos]=CUDA_CANNY_LOW_COLOR;
    else if(preResult==2)
        imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
    else // Weak edge; decide whether to keep it (if at least one neighboring pixel is a strong edge)
    {
        if(hasLeft)
        {
            if(blockPreResultCache[blockPreResultCachePos-1]==2) // preResult[y*width+leftX]==1.0f
            {
                imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
                return;
            }
            if(hasTop)
            {
                if(blockPreResultCache[blockPreResultCachePos-blockPreResultCacheRowSize]==2) // preResult[topY*width+leftX]==1.0f
                {
                    imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
                    return;
                }
            }
            if(hasBottom)
            {
                if(blockPreResultCache[blockPreResultCachePos+blockPreResultCacheRowSize]==2) // preResult[bottomY*width+leftX]==1.0f
                {
                    imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
                    return;
                }
            }
        }
        if(hasRight)
        {
            if(blockPreResultCache[blockPreResultCachePos+1]==2) // preResult[y*width+rightX]==1.0f
            {
                imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
                return;
            }
            if(hasTop)
            {
                if(blockPreResultCache[blockPreResultCachePos-blockPreResultCacheRowSize+1]==2) // preResult[topY*width+rightX]==1.0f
                {
                    imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
                    return;
                }
            }
            if(hasBottom)
            {
                if(blockPreResultCache[blockPreResultCachePos+blockPreResultCacheRowSize+1]==2) // preResult[bottomY*width+rightX]==1.0f
                {
                    imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
                    return;
                }
            }
        }
        if(hasTop)
        {
            if(blockPreResultCache[blockPreResultCachePos-blockPreResultCacheRowSize]==2) // preResult[topY*width+x]==1.0f
            {
                imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
                return;
            }
        }
        if(hasBottom)
        {
            if(blockPreResultCache[blockPreResultCachePos+blockPreResultCacheRowSize]==2) // preResult[bottomY*width+x]==1.0f
            {
                imageDataOut[pos]=CUDA_CANNY_HIGH_COLOR;
                return;
            }
        }
        imageDataOut[pos]=CUDA_CANNY_LOW_COLOR;
    }
}

__global__ void device_sobelEdgeDetect(uint32_t *imageData,int width,int height,float amplifier,uint32_t *imageDataOut)
{
    __shared__ float blockImageSection[(2+CUDA_BLOCK_SIDE_LENGTH)*(2+CUDA_BLOCK_SIDE_LENGTH)]; // Left/right/top/bottom pixel rows included

    int blockHorId=blockIdx.x;
    int blockVerId=blockIdx.y;
    int relativeX=threadIdx.x;
    int x=blockHorId*blockDim.x+relativeX;
    int relativeY=threadIdx.y;
    int y=blockVerId*blockDim.y+relativeY;

    if(x>=width||y>=height)
        return;

    size_t pos=y*width+x;
    uint32_t color=imageData[pos];

    // Fill up blockImageSection with B/W versions of pixels

    // This pixel:
    size_t blockImageSectionPos=(1+relativeY)*(CUDA_BLOCK_SIDE_LENGTH+2)+(1+relativeX); // "1+...": border pixels
    float component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
    blockImageSection[blockImageSectionPos]=component;

    bool leftmostInBlock=false;
    bool rightmostInBlock=false;
    bool topmostInBlock=false;
    bool bottommostInBlock=false;

    bool leftmostInImage;
    bool topmostInImage;
    bool rightmostInImage=x==width-1;
    bool bottommostInImage=y==height-1;

    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
    // avoid false positives on the borders of the image

    if(rightmostInBlock=(relativeX==CUDA_BLOCK_SIDE_LENGTH-1||rightmostInImage))
    {
        if(rightmostInImage)
        {
            color=imageData[pos];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos+1]=component;
        }
        else
        {
            color=imageData[pos+1];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos+1]=component;
        }
    }
    else if(leftmostInBlock=(relativeX==0))
    {
        if(leftmostInImage=(x==0))
        {
            color=imageData[pos];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos-1]=component;
        }
        else
        {
            color=imageData[pos-1];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos-1]=component;
        }
    }

    if(bottommostInBlock=(relativeY==CUDA_BLOCK_SIDE_LENGTH-1||bottommostInImage))
    {
        if(bottommostInImage)
        {
            color=imageData[pos];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)]=component;
            if(leftmostInBlock)
            {
                if(leftmostInImage)
                {
                    //color=imageData[pos]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)-1]=component;
                }
                else
                {
                    color=imageData[pos-1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)-1]=component;
                }
            }
            else if(rightmostInBlock)
            {
                if(rightmostInImage)
                {
                    //color=imageData[pos]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)+1]=component;
                }
                else
                {
                    color=imageData[pos+1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)+1]=component;
                }
            }
        }
        else
        {
            color=imageData[pos+width];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)]=component;
            if(leftmostInBlock)
            {
                if(leftmostInImage)
                {
                    //color=imageData[pos+width]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)-1]=component;
                }
                else
                {
                    color=imageData[pos+width-1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)-1]=component;
                }
            }
            else if(rightmostInBlock)
            {
                if(rightmostInImage)
                {
                    //color=imageData[pos+width]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)+1]=component;
                }
                else
                {
                    color=imageData[pos+width+1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+2)+1]=component;
                }
            }
        }
    }
    else if(topmostInBlock=(relativeY==0))
    {
        if(topmostInImage=(y==0))
        {
            color=imageData[pos];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)]=component;
            if(leftmostInBlock)
            {
                if(leftmostInImage)
                {
                    //color=imageData[pos]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)-1]=component;
                }
                else
                {
                    color=imageData[pos-1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)-1]=component;
                }
            }
            else if(rightmostInBlock)
            {
                if(rightmostInImage)
                {
                    //color=imageData[pos]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)+1]=component;
                }
                else
                {
                    color=imageData[pos+1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)+1]=component;
                }
            }
        }
        else
        {
            color=imageData[pos-width];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)]=component;
            if(leftmostInBlock)
            {
                if(leftmostInImage)
                {
                    //color=imageData[pos-width]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)-1]=component;
                }
                else
                {
                    color=imageData[pos-width-1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)-1]=component;
                }
            }
            else if(rightmostInBlock)
            {
                if(rightmostInImage)
                {
                    //color=imageData[pos-width]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)+1]=component;
                }
                else
                {
                    color=imageData[pos-width+1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos-(CUDA_BLOCK_SIDE_LENGTH+2)+1]=component;
                }
            }
        }
    }

    __syncthreads();

    float accXValueSum=0.0f;
    float accYValueSum=0.0f;

    // Left/top

    // X accumulator
    accXValueSum+=blockImageSection[(1+relativeY-1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX-1]*(-1.0f);
    // Y accumulator
    accYValueSum+=blockImageSection[(1+relativeY-1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX-1]*(-1.0f);

    // Left/bottom

    // X accumulator
    accXValueSum+=blockImageSection[(1+relativeY+1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX-1]*(-1.0f);
    // Y accumulator
    accYValueSum+=blockImageSection[(1+relativeY+1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX-1]*(1.0f);

    // Left

    // X accumulator
    accXValueSum+=blockImageSection[(1+relativeY)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX-1]*(-2.0f);

    // Right/top

    // X accumulator
    accXValueSum+=blockImageSection[(1+relativeY-1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX+1]*(1.0f);
    // Y accumulator
    accYValueSum+=blockImageSection[(1+relativeY-1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX+1]*(-1.0f);

    // Right/bottom

    // X accumulator
    accXValueSum+=blockImageSection[(1+relativeY+1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX+1]*(1.0f);
    // Y accumulator
    accYValueSum+=blockImageSection[(1+relativeY+1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX+1]*(1.0f);

    // Right

    // X accumulator
    accXValueSum+=blockImageSection[(1+relativeY)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX+1]*(2.0f);

    // Top

    // Y accumulator
    accYValueSum+=blockImageSection[(1+relativeY-1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX]*(-2.0f);

    // Bottom

    // Y accumulator
    accYValueSum+=blockImageSection[(1+relativeY+1)*(CUDA_BLOCK_SIDE_LENGTH+2)+1+relativeX]*(2.0f);

    // Final result

    component=amplifier*sqrt(accXValueSum*accXValueSum+accYValueSum*accYValueSum);
    uint8_t componentOut=(uint8_t)round(component*255.0);
    imageDataOut[pos]=getColor(255,componentOut,componentOut,componentOut);
}

__global__ void device_robertsEdgeDetect(uint32_t *imageData,int width,int height,float amplifier,uint32_t *imageDataOut)
{
    __shared__ float blockImageSection[(1+CUDA_BLOCK_SIDE_LENGTH)*(1+CUDA_BLOCK_SIDE_LENGTH)]; // Right/bottom pixel rows included

    int blockHorId=blockIdx.x;
    int blockVerId=blockIdx.y;
    int relativeX=threadIdx.x;
    int x=blockHorId*blockDim.x+relativeX;
    int relativeY=threadIdx.y;
    int y=blockVerId*blockDim.y+relativeY;

    if(x>=width||y>=height)
        return;

    int pos=y*width+x;
    uint32_t color=imageData[pos];

    // Fill up blockImageSection with B/W versions of pixels

    // Exact same pixel:
    size_t blockImageSectionPos=relativeY*(CUDA_BLOCK_SIDE_LENGTH+1)+relativeX; // "1+...": border pixels
    float component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
    blockImageSection[blockImageSectionPos]=component;

    bool rightmostInBlock=false;
    bool bottommostInBlock=false;

    bool rightmostInImage=x==width-1;
    bool bottommostInImage=y==height-1;

    // Extend the image by 1 pixel on each side and using the outmost pixels in order to
    // avoid false positives on the borders of the image

    if(rightmostInBlock=(relativeX==CUDA_BLOCK_SIDE_LENGTH-1||rightmostInImage))
    {
        if(rightmostInImage)
        {
            color=imageData[pos];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos+1]=component;
        }
        else
        {
            color=imageData[pos+1];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos+1]=component;
        }
    }

    if(bottommostInBlock=(relativeY==CUDA_BLOCK_SIDE_LENGTH-1||bottommostInImage))
    {
        if(bottommostInImage)
        {
            color=imageData[pos];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+1)]=component;
            if(rightmostInBlock)
            {
                if(rightmostInImage)
                {
                    //color=imageData[pos]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+1)+1]=component;
                }
                else
                {
                    color=imageData[pos+1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+1)+1]=component;
                }
            }
        }
        else
        {
            color=imageData[pos+width];
            component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
            blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+1)]=component;
            if(rightmostInBlock)
            {
                if(rightmostInImage)
                {
                    //color=imageData[pos+width]; // Already set
                    //component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+1)+1]=component;
                }
                else
                {
                    color=imageData[pos+width+1];
                    component=getFBWComponent(getFRed(color),getFGreen(color),getFBlue(color));
                    blockImageSection[blockImageSectionPos+(CUDA_BLOCK_SIDE_LENGTH+1)+1]=component;
                }
            }
        }
    }

    __syncthreads();

    float accXValueSum=0.0f;
    float accYValueSum=0.0f;

    // Right/bottom

    // X accumulator
    accXValueSum+=blockImageSection[(relativeY+1)*(CUDA_BLOCK_SIDE_LENGTH+1)+relativeX+1]*(-1.0f);

    // Right

    // Y accumulator
    accYValueSum+=blockImageSection[(relativeY)*(CUDA_BLOCK_SIDE_LENGTH+1)+relativeX+1]*(1.0f);

    // Bottom

    // Y accumulator
    accYValueSum+=blockImageSection[(relativeY+1)*(CUDA_BLOCK_SIDE_LENGTH+1)+relativeX]*(-1.0f);

    // This pixel:
    // X accumulator
    accXValueSum+=blockImageSection[(relativeY)*(CUDA_BLOCK_SIDE_LENGTH+1)+relativeX]*(1.0f);

    // Final result

    component=amplifier*sqrt(accXValueSum*accXValueSum+accYValueSum*accYValueSum);
    uint8_t componentOut=(uint8_t)round(component*255.0);
    imageDataOut[pos]=getColor(255,componentOut,componentOut,componentOut);
}

#define device_cudaGaussianBlur_imageDataPos(x) (4*(x)+channel)

__global__ void device_cudaGaussianBlur(uint8_t *imageData,int width,int height,int filterSize,float deviation,uint8_t *imageDataOut)
{
    const int blockImageSectionSingleChannelSizeSqrt=CUDA_3_CHANNEL_MAX_FILTER_SIZE*2+CUDA_GAUSSIAN_BLUR_BLOCK_SIDE_LENGTH;
    __shared__ float blockImageSection[blockImageSectionSingleChannelSizeSqrt*blockImageSectionSingleChannelSizeSqrt];

    int filterSizeInPixels=2*filterSize+1;
    int channel=blockIdx.z; // Only needed if on big-endian machine: add 1 for the alpha channel (to simplify memory ops)
    int blockHorId=blockIdx.x;
    int blockVerId=blockIdx.y;
    int relativeX=threadIdx.x;
    int x=blockHorId*blockDim.x+relativeX;
    int relativeY=threadIdx.y;
    int y=blockVerId*blockDim.y+relativeY;

    int rowSize=2*filterSize+blockDim.x;
    int columnSize=2*filterSize+blockDim.y;
    int topOffsetRows=filterSize;

    if(x>=width||y>=height)
        return;

    size_t pixelId=y*width+x;
    size_t pos=sizeof(uint32_t)*pixelId+channel;
    uint8_t component=imageData[pos];
    float color=colorComponentToF(component);

    // Copy pixels

    // Copy this pixel from source image

    int posInBlockImageSectionData=(topOffsetRows+relativeY)*rowSize+filterSize+relativeX;
    blockImageSection[posInBlockImageSectionData]=color;

    // Copy extra pixels, if needed

    int relativeLeftExtra=-filterSize+relativeX;
    int relativeRightExtra=relativeX+filterSize;
    int relativeTopExtra=-filterSize+relativeY;
    int relativeBottomExtra=relativeY+filterSize;

    int effectiveBlockWidth=blockHorId==gridDim.x-1?(width-blockHorId*blockDim.x):blockDim.x;
    int effectiveBlockHeight=blockVerId==gridDim.y-1?(height-blockVerId*blockDim.y):blockDim.y;

    bool copyExtraFromLeft=false;
    bool leftOverflow=false;
    bool copyExtraFromRight=false;
    bool rightOverflow=false;
    int absLeftExtra;
    int absRightExtra;

    if(copyExtraFromLeft=(relativeLeftExtra<0))
    {
        absLeftExtra=x-filterSize;
        if(leftOverflow=(absLeftExtra<0))
            blockImageSection[posInBlockImageSectionData-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(y*width)]);
        else
            blockImageSection[posInBlockImageSectionData-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(y*width+absLeftExtra)]);
    }
    if(copyExtraFromRight=(relativeRightExtra>=effectiveBlockWidth))
    {
        absRightExtra=x+filterSize;
        if(rightOverflow=(absRightExtra>=width))
            blockImageSection[posInBlockImageSectionData+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(y*width+width-1)]);
        else
            blockImageSection[posInBlockImageSectionData+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(y*width+absRightExtra)]);
    }
    if(relativeTopExtra<0)
    {
        int absTopExtra=y-filterSize;
        if(absTopExtra<0)
        {
            blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((-absTopExtra)*width+x)]);
            if(copyExtraFromLeft)
            {
                if(leftOverflow)
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((-absTopExtra)*width)]);
                else
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((-absTopExtra)*width+absLeftExtra)]);
            }
            if(copyExtraFromRight)
            {
                if(rightOverflow)
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((-absTopExtra)*width+width-1)]);
                else
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((-absTopExtra)*width+absRightExtra)]);
            }
        }
        else
        {
            blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absTopExtra*width+x)]);
            if(copyExtraFromLeft)
            {
                if(leftOverflow)
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absTopExtra*width)]);
                else
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absTopExtra*width+absLeftExtra)]);
            }
            if(copyExtraFromRight)
            {
                if(rightOverflow)
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absTopExtra*width+width-1)]);
                else
                    blockImageSection[posInBlockImageSectionData-(filterSize*rowSize)+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absTopExtra*width+absRightExtra)]);
            }
        }
    }
    if(relativeBottomExtra>=effectiveBlockHeight)
    {
        int absBottomExtra=y+filterSize;
        if(absBottomExtra>=height)
        {
            blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((height-(absBottomExtra-height+1))*width+x)]);
            if(copyExtraFromLeft)
            {
                if(leftOverflow)
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((height-(absBottomExtra-height+1))*width)]);
                else
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((height-(absBottomExtra-height+1))*width+absLeftExtra)]);
            }
            if(copyExtraFromRight)
            {
                if(rightOverflow)
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((height-(absBottomExtra-height+1))*width+width-1)]);
                else
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos((height-(absBottomExtra-height+1))*width+absRightExtra)]);
            }
        }
        else
        {
            blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absBottomExtra*width+x)]);
            if(copyExtraFromLeft)
            {
                if(leftOverflow)
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absBottomExtra*width)]);
                else
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)-filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absBottomExtra*width+absLeftExtra)]);
            }
            if(copyExtraFromRight)
            {
                if(rightOverflow)
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absBottomExtra*width+width-1)]);
                else
                    blockImageSection[posInBlockImageSectionData+(filterSize*rowSize)+filterSize]=colorComponentToF(imageData[device_cudaGaussianBlur_imageDataPos(absBottomExtra*width+absRightExtra)]);
            }
        }
    }

    float preFactor=(1.0f/(2.0f*M_PI_F*deviation*deviation));
    float preFactor2=(2.0f*deviation*deviation);
    float filterFactors[CUDA_3_CHANNEL_MAX_FILTER_SIZE*CUDA_3_CHANNEL_MAX_FILTER_SIZE];
    for(int filterY=0;filterY<filterSizeInPixels;filterY++)
    {
        int offset=filterY*filterSizeInPixels;
        for(int filterX=0;filterX<filterSizeInPixels;filterX++)
        {
            int n1=filterX+1-filterSize-1;
            int n2=filterY+1-filterSize-1;
            float factor=preFactor*exp(-(((float)(n1*n1+n2*n2))/preFactor2));
            filterFactors[offset+filterX]=factor;
        }
    }

    __syncthreads();

    int relativeXWithBorderPixels=relativeX+filterSize;
    int relativeYWithBorderPixels=relativeY+filterSize;
    float pixelValueSum=0.0f;

    for(int yOfFilter=0;yOfFilter<filterSizeInPixels;yOfFilter++)
    {
        int yWithFilter=-filterSize+relativeYWithBorderPixels+yOfFilter;
         // Use symmetry to compensate for missing pixels (in order to avoid dark borders)
        if(yWithFilter<0)
            yWithFilter=(relativeYWithBorderPixels+filterSize)-yOfFilter;
        else if(yWithFilter>=columnSize)
            yWithFilter=(relativeYWithBorderPixels-filterSize)+(filterSizeInPixels-1-yOfFilter);
        for(int xOfFilter=0;xOfFilter<filterSizeInPixels;xOfFilter++)
        {
            int xWithFilter=-filterSize+relativeXWithBorderPixels+xOfFilter;
            // Use symmetry to compensate for missing pixels (in order to avoid dark borders)
            if(xWithFilter<0)
                xWithFilter=(relativeXWithBorderPixels+filterSize)-xOfFilter;
            else if(xWithFilter>=rowSize)
                xWithFilter=(relativeXWithBorderPixels-filterSize)+(filterSizeInPixels-1-xOfFilter);

            float factor=filterFactors[yOfFilter*filterSizeInPixels+xOfFilter];
            pixelValueSum+=blockImageSection[yWithFilter*rowSize+xWithFilter]*factor;
        }
    }

    imageDataOut[pos]=colorFToComponent(pixelValueSum);
    if(channel==0)
        imageDataOut[pos+3]=0xff;
}

#undef device_cudaGaussianBlur_imageDataPos

uint32_t *cudaGetBWImage(uint32_t *imageData,int width,int height)
{
    int blocksPerLine=ceil(floatDiv(width,CUDA_THREADS_PER_BLOCK));
    int totalNumBlocks=blocksPerLine*height; // Each block is responsible for a single line

    if(totalNumBlocks>65535)
        return 0;

    uint32_t *device_origImageData_in;
    uint32_t *device_newImageData_out;
    size_t imageSize=width*height*sizeof(uint32_t);
    cudaMalloc(&device_origImageData_in,imageSize);
    cudaMalloc(&device_newImageData_out,imageSize);

    cudaMemcpy(device_origImageData_in,imageData,imageSize,cudaMemcpyHostToDevice);

    dim3 blocks(blocksPerLine,height,1);

    device_getBWImage<<<blocks,CUDA_THREADS_PER_BLOCK>>>(device_origImageData_in,width,device_newImageData_out);

    uint32_t *newImageData=(uint32_t*)malloc(imageSize);
    cudaMemcpy(newImageData,device_newImageData_out,imageSize,cudaMemcpyDeviceToHost);

    cudaFree(device_newImageData_out);
    cudaFree(device_origImageData_in);

    return newImageData;
}

uint32_t *cudaCannyEdgeDetect(uint32_t *imageData,int width,int height,float deviation,float highTreshold,float lowTreshold)
{
    // Do not first convert to BW, as that would force us to copy memory back and forth

    // Do not use CUDA_THREADS_PER_BLOCK here!

    int horBlockCount=ceil(floatDiv(width,CUDA_BLOCK_SIDE_LENGTH));
    int verBlockCount=ceil(floatDiv(height,CUDA_BLOCK_SIDE_LENGTH));
    int totalNumBlocks=horBlockCount*verBlockCount;

    if(totalNumBlocks>65535)
        return 0;

    uint32_t *device_origImageData_in;
    uint32_t *device_newImageData_out;
    float *device_gaussianBwArray;
    float *device_gradientArray;
    float *device_gradientAtan2Array;
    uint8_t *device_preResultArray;
    size_t imageArea=width*height;
    size_t imageSize=imageArea*sizeof(uint32_t);
    size_t imageSizeFloat=imageArea*sizeof(float);
    cudaMalloc(&device_origImageData_in,imageSize);
    cudaMalloc(&device_newImageData_out,imageSize);
    cudaMalloc(&device_gaussianBwArray,imageSizeFloat);
    cudaMalloc(&device_gradientArray,imageSizeFloat);
    cudaMalloc(&device_gradientAtan2Array,imageSizeFloat);
    cudaMalloc(&device_preResultArray,imageArea*sizeof(uint8_t));

    cudaMemcpy(device_origImageData_in,imageData,imageSize,cudaMemcpyHostToDevice);

    dim3 blocks(horBlockCount,verBlockCount,1);
    dim3 threadsPerBlock(CUDA_BLOCK_SIDE_LENGTH,CUDA_BLOCK_SIDE_LENGTH,1);

    int filterSize=floor(deviation*3.0f); // NVidia standard

    device_cannyEdgeDetect_stage1<<<blocks,threadsPerBlock>>>(device_origImageData_in,width,height,filterSize,deviation,device_gaussianBwArray);
    device_cannyEdgeDetect_stage2<<<blocks,threadsPerBlock>>>(width,height,filterSize,device_gaussianBwArray,device_gradientArray,device_gradientAtan2Array);
    device_cannyEdgeDetect_stage3<<<blocks,threadsPerBlock>>>(width,height,highTreshold,lowTreshold,device_gradientArray,device_gradientAtan2Array,device_preResultArray);
    device_cannyEdgeDetect_stage4<<<blocks,threadsPerBlock>>>(width,height,device_preResultArray,device_newImageData_out);

    uint32_t *newImageData=(uint32_t*)malloc(imageSize);
    cudaMemcpy(newImageData,device_newImageData_out,imageSize,cudaMemcpyDeviceToHost);

    cudaFree(device_preResultArray);
    cudaFree(device_gradientAtan2Array);
    cudaFree(device_gradientArray);
    cudaFree(device_gaussianBwArray);
    cudaFree(device_newImageData_out);
    cudaFree(device_origImageData_in);

    return newImageData;
}

uint32_t *cudaSobelEdgeDetect(uint32_t *imageData,int width,int height,float amplifier)
{
    // Do not first convert to BW, as that would force us to copy memory back and forth

    // Do not use CUDA_THREADS_PER_BLOCK here!

    int horBlockCount=ceil(floatDiv(width,/*!!!*/CUDA_BLOCK_SIDE_LENGTH));
    int verBlockCount=ceil(floatDiv(height,/*!!!*/CUDA_BLOCK_SIDE_LENGTH));
    int totalNumBlocks=horBlockCount*verBlockCount;

    if(totalNumBlocks>65535)
        return 0;

    uint32_t *device_origImageData_in;
    uint32_t *device_newImageData_out;
    size_t imageSize=width*height*sizeof(uint32_t);
    cudaMalloc(&device_origImageData_in,imageSize);
    cudaMalloc(&device_newImageData_out,imageSize);

    cudaMemcpy(device_origImageData_in,imageData,imageSize,cudaMemcpyHostToDevice);

    dim3 blocks(horBlockCount,verBlockCount,1);
    dim3 threadsPerBlock(CUDA_BLOCK_SIDE_LENGTH,CUDA_BLOCK_SIDE_LENGTH,1);

    device_sobelEdgeDetect<<<blocks,threadsPerBlock>>>(device_origImageData_in,width,height,amplifier,device_newImageData_out);

    uint32_t *newImageData=(uint32_t*)malloc(imageSize);
    cudaMemcpy(newImageData,device_newImageData_out,imageSize,cudaMemcpyDeviceToHost);

    cudaFree(device_newImageData_out);
    cudaFree(device_origImageData_in);

    return newImageData;
}

uint32_t *cudaRobertsEdgeDetect(uint32_t *imageData,int width,int height,float amplifier)
{
    // Do not first convert to BW, as that would force us to copy memory back and forth

    // Do not use CUDA_THREADS_PER_BLOCK here!

    int horBlockCount=ceil(floatDiv(width,/*!!!*/CUDA_BLOCK_SIDE_LENGTH));
    int verBlockCount=ceil(floatDiv(height,/*!!!*/CUDA_BLOCK_SIDE_LENGTH));
    int totalNumBlocks=horBlockCount*verBlockCount;

    if(totalNumBlocks>65535)
        return 0;

    uint32_t *device_origImageData_in;
    uint32_t *device_newImageData_out;
    size_t imageSize=width*height*sizeof(uint32_t);
    cudaMalloc(&device_origImageData_in,imageSize);
    cudaMalloc(&device_newImageData_out,imageSize);

    cudaMemcpy(device_origImageData_in,imageData,imageSize,cudaMemcpyHostToDevice);

    dim3 blocks(horBlockCount,verBlockCount,1);
    dim3 threadsPerBlock(CUDA_BLOCK_SIDE_LENGTH,CUDA_BLOCK_SIDE_LENGTH,1);

    device_robertsEdgeDetect<<<blocks,threadsPerBlock>>>(device_origImageData_in,width,height,amplifier,device_newImageData_out);

    uint32_t *newImageData=(uint32_t*)malloc(imageSize);
    cudaMemcpy(newImageData,device_newImageData_out,imageSize,cudaMemcpyDeviceToHost);

    cudaFree(device_newImageData_out);
    cudaFree(device_origImageData_in);

    return newImageData;
}

uint32_t *cuda3ChannelGaussianBlur(uint32_t *imageData,int width,int height,int filterSize,float deviation)
{
    // Do not first convert to BW, as that would force us to copy memory back and forth

    // Do not use CUDA_THREADS_PER_BLOCK here!

    int horBlockCount=ceil(floatDiv(width,/*!!!*/CUDA_GAUSSIAN_BLUR_BLOCK_SIDE_LENGTH));
    int verBlockCount=ceil(floatDiv(height,/*!!!*/CUDA_GAUSSIAN_BLUR_BLOCK_SIDE_LENGTH));
    int totalNumBlocks=horBlockCount*verBlockCount;

    if(totalNumBlocks*3>65535||filterSize>CUDA_3_CHANNEL_MAX_FILTER_SIZE)
        return 0;

    uint8_t *device_origImageData_in;
    uint8_t *device_newImageData_out;
    size_t imageArea=width*height;
    size_t imageSize=imageArea*sizeof(uint32_t);
    cudaMalloc(&device_origImageData_in,imageSize);
    cudaMalloc(&device_newImageData_out,imageSize);

    cudaMemcpy(device_origImageData_in,imageData,imageSize,cudaMemcpyHostToDevice);

    dim3 blocks(horBlockCount,verBlockCount,3); // 3: R, G and B channel
    dim3 threadsPerBlock(CUDA_GAUSSIAN_BLUR_BLOCK_SIDE_LENGTH,CUDA_GAUSSIAN_BLUR_BLOCK_SIDE_LENGTH,1);

    device_cudaGaussianBlur<<<blocks,threadsPerBlock>>>(device_origImageData_in,width,height,filterSize,deviation,device_newImageData_out);

    uint32_t *newImageData=(uint32_t*)malloc(imageSize);
    cudaMemcpy(newImageData,device_newImageData_out,imageSize,cudaMemcpyDeviceToHost);

    // Alpha channel filled in device_cudaGaussianBlur
    //for(size_t i=0;i<imageArea;i++) // Fill alpha channel
    //    newImageData[i]|=0xff000000;

    cudaFree(device_newImageData_out);
    cudaFree(device_origImageData_in);

    return newImageData;
}
