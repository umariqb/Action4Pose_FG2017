#include <stdlib.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv/cxcore.hpp>
#include "armaknn.h"

using namespace std;

void inline rgb2lab( const arma::Cube<unsigned char> &oRGB,  arma::Cube<float> &oLAB)
{
  //convert to float and extract channels
  arma::fmat oR = arma::conv_to<arma::fmat>::from( oRGB.slice(0) );
  arma::fmat oG = arma::conv_to<arma::fmat>::from( oRGB.slice(1) );
  arma::fmat oB = arma::conv_to<arma::fmat>::from( oRGB.slice(2) );

  //make sure to normalize to 1.0
  if ((max(max(oR)) > 1.0) | (max(max(oG)) > 1.0) | (max(max(oB)) > 1.0))
  {
    oR = oR/255.0;
    oG = oG/255.0;
    oB = oB/255.0;
  }

  //get data
  const unsigned int M = oR.n_rows;
  const unsigned int N = oR.n_cols;
  const unsigned int s = M*N;

  //Set a threshold
  const float T = 0.008856;

  arma::fmat oRGB2(3, s);
  oRGB2.row(0) = reshape(oR, 1, s);
  oRGB2.row(1) = reshape(oG, 1, s);
  oRGB2.row(2) = reshape(oB, 1, s);


  //RGB to XYZ
  arma::fmat oMAT;
  oMAT << 0.412453 << 0.357580 << 0.180423 <<arma::endr <<
          0.212671 << 0.715160 << 0.072169 <<arma::endr <<
          0.019334 << 0.119193 << 0.950227 <<arma::endr;

  arma::fmat XYZ = oMAT * oRGB2;

  arma::fmat X = XYZ.row(0) / 0.950456;
  arma::fmat Y = XYZ.row(1);
  arma::fmat Z = XYZ.row(2) / 1.088754;

  arma::umat XT = X > T;
  arma::umat NXT = X <= T;
  arma::umat YT = Y > T;
  arma::umat NYT = Y <= T;
  arma::umat ZT = Z > T;
  arma::umat NZT = Z <= T;

  arma::fmat fX = XT % arma::pow(X, (1.0/3.0)) + (NXT) % (7.787 * X + 16/116);

  //Compute L
  arma::fmat Y3 = pow(Y,(1.0/3.0));
  arma::fmat fY = YT % Y3 + (NYT) % (7.787 * Y + 16/116);
  arma::fmat L  = YT % (116 * Y3 - 16.0) + (NYT) % (903.3 * Y);
  arma::fmat fZ = ZT % arma::pow(Z, (1.0/3.0)) + (NZT) % (7.787 * Z + 16/116);

  //Compute a and b
  arma::fmat a = 500 * (fX - fY);
  arma::fmat b = 200 * (fY - fZ);

  oLAB.slice(0) = reshape(L, M, N);
  oLAB.slice(1) = reshape(a, M, N);
  oLAB.slice(2) = reshape(b, M, N);
}


//

int main(int argc, char** argv) {

  //global parameters
  const unsigned int knn = 1; //no soft assignement	
  const unsigned int maxpixels = 128*128;

  std::string image_file;
  std::string centroids_file;

  //program options
  try {
    boost::program_options::options_description desc("Allowed options");
    boost::program_options::variables_map vm;
    {
      using namespace boost::program_options;
      desc.add_options()
      ("help",        "Produce help message")
      ("image",    	   value<std::string>(&image_file)->default_value(""),  "The image to process")
      ("centroids",    value<std::string>(&centroids_file)->default_value(""),  "The vocabulary file (arma raw format)")
      ;
    }
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    boost::program_options::notify(vm);
    if (vm.count("help") || vm.count("image")==0 || vm.count("centroids")==0 ){
      cout << "Usage: options_description [options]\n";
      cout << desc;
      return 0;
    }
  } catch(std::exception& e) {
    cout << "Error reading options: " <<e.what() << endl;
    return -1;
  }

  //read voabulary
  arma::fmat centroids;
  centroids.load(centroids_file, arma::raw_ascii);

  //Load image
  cv::Mat BGR_image;
  BGR_image = cv::imread( image_file.c_str(), 1 );
  cv::Size s = BGR_image.size();

  //resize image
  float scale = (float)maxpixels/s.width/s.height;
  scale=sqrt(scale);
  s.width *= scale;
  s.height *= scale;
  cv::resize( BGR_image,  BGR_image, s);

  //copy into armadillo somewhta dirrrty, but works at least compared to openCV cvtColor
  arma::Cube<unsigned char> oImage(s.width, s.height, 3);
  for (int i = 0; i < BGR_image.rows; i++) {
    for (int j = 0; j < BGR_image.cols; j++) {
      oImage(j, i, 0) = BGR_image.at<cv::Vec3b>(i, j)[2];
      oImage(j, i, 1) = BGR_image.at<cv::Vec3b>(i, j)[1];
      oImage(j, i, 2) = BGR_image.at<cv::Vec3b>(i, j)[0];
    }
  }

  //convert L*a*b*
  arma::Cube<float> oLab(oImage.n_rows, oImage.n_cols, oImage.n_slices );
  rgb2lab(oImage, oLab);

  //some shortcuts
  const unsigned int w = oImage.n_cols;
  const unsigned int h = oImage.n_rows;

  //create large vector array of colors
  arma::fmat L = arma::reshape(oLab.slice(0), 1, w*h);
  arma::fmat a = arma::reshape(oLab.slice(1), 1, w*h);
  arma::fmat b = arma::reshape(oLab.slice(2), 1, w*h);
  arma::fmat all_pixels = arma::join_cols(arma::join_cols(L, a), b);

  //get nearest neighbors
  arma::fmat distanceMap;
  arma::umat indexMap;
  knn_L2( all_pixels,
          centroids,
          distanceMap,
          indexMap,
          knn);

  //init vector
  arma::fvec boc;
  boc = arma::zeros<fvec>(centroids.n_cols);

  //make histogram
  for(unsigned int i=0; i<all_pixels.n_cols;i++) {
    for(unsigned int k=0;k<knn;k++){ //multi assignment/soft assignment
      const unsigned int idx = indexMap(k,i);
      boc(idx) += 1;
    }
  }

  //powerlaw
  boc = arma::sqrt(boc);

  //renormalize L1
  float norm = arma::sum(boc);
  boc *= 1.0f/norm;

  //TODO: do a PCA

  //print
  arma::trans(boc).print("boc");

  //ciao
  return (EXIT_SUCCESS);
}

