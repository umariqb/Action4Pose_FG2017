/***************************************************************************
 *   Copyright (C) 2006 by Mian Zhou   *
 *   M.Zhou@reading.ac.uk   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef CVGABOR_H
#define CVGABOR_H

#include <iostream>


#include <opencv/cv.h>
#include <opencv/highgui.h>


#ifdef USE_TBB
#include "tbb/tbb.h"
#endif



/**
@author Mian Zhou
*/
class CvGabor{
public:
    CvGabor();
    ~CvGabor();


     CvGabor(int iMu, int iNu, double dSigma);
     CvGabor(int iMu, int iNu, double dSigma, double dF);
     CvGabor(double dPhi, int iNu);
     CvGabor(double dPhi, int iNu, double dSigma);
     CvGabor(double dPhi, int iNu, double dSigma, double dF);
    bool IsInit();
    long mask_width();
    IplImage* get_image(int Type);
    bool IsKernelCreate();
    long get_mask_width();
    void Init(int iMu, int iNu, double dSigma, double dF);
    void Init(double dPhi, int iNu, double dSigma, double dF);
    void output_file(const char *filename, int Type);
    CvMat* get_matrix(int Type);
    void show(int Type);
    void conv_img(const IplImage *src, IplImage *dst, int Type);
    void conv_img(const cv::Mat& src, cv::Mat& dst, int Type);
     CvGabor(int iMu, int iNu);
    void normalize( const CvArr* src, CvArr* dst, double a, double b, int norm_type, const CvArr* mask );
    void conv_img_a(IplImage *src, IplImage *dst, int Type);

protected:
    double Sigma;
    double F;
    double Kmax;
    double K;
    double Phi;
    bool bInitialised;
    bool bKernel;
    long Width;
    CvMat *Imag;
    CvMat *Real;

private:
    void creat_kernel();
};


class GaborConverter
{

public:
    GaborConverter( const cv::Mat& img_, std::vector<cv::Mat>& vImg_, int offSet_, bool useIntegral_ ):
        img(img_), vImg(vImg_), offSet( offSet_ ), useIntegral(useIntegral_)
    {};

#ifdef USE_TBB
    void operator() (const tbb::blocked_range<size_t>& r) const
    {
        for (size_t stripe=r.begin(); stripe!=r.end(); ++stripe)
            calcGabor(stripe);
    }
#endif

    void convert()
    {
        for (int i=0; i < 4*7; i++)
        {
            calcGabor(i);
        }
    }

private:

    void calcGabor( int i) const
    {
        const double PI = 3.14159265;
        int NuMin = 0;
        int NuMax = 4;
        int MuMin = 0;
        int MuMax = 7;
        double sigma = 1./2.0 * PI;
        double dF = sqrt(2.0);

        int iMu = 0;
        int iNu = 0;


        //TODO
        int j=0;
        bool f = false;
        for (iNu = NuMin; iNu <= NuMax; iNu++)
        {
            for(iMu = MuMin; iMu < MuMax; iMu++)
            {
                if( j == i)
                {
                    f = true;
                    break;
                }
                j++;

            }
            if (f) break;
        }

        cv::Mat gabor_img = cv::Mat(img.size(),CV_8U);

        CvGabor gabor(iMu, iNu, sigma , dF);
        gabor.conv_img(img,  gabor_img, 3);

        if( useIntegral )
        {
            cv::Mat int_dest;
            integral( gabor_img, int_dest, CV_32F);
            vImg[offSet+i] = int_dest;

        }else{
            vImg[offSet+i] = gabor_img;
        }
    }

    const cv::Mat& img;
    std::vector<cv::Mat>& vImg;
    int offSet;
    bool useIntegral;
};

#endif
