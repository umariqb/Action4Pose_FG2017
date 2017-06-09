#ifndef VOTMAPMAXIMA_H
#define VOTMAPMAXIMA_H

#include <opencv2/opencv.hpp>
#include <boost/serialization/utility.hpp>


namespace learning {
namespace forest {

class VotMapMaxCoords
{
  public:
    VotMapMaxCoords(){};
    VotMapMaxCoords(cv::Mat_<int> coords_x_, cv::Mat_<int> coords_y_):
                coords_x(coords_x_), coords_y(coords_y_)
                {};

    cv::Mat_<int> coords_x;
    cv::Mat_<int> coords_y;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & coords_x;
        ar & coords_y;
    }
};

bool save_votmap_max_coords(VotMapMaxCoords& coords, std::string& fname);
bool save_votmap_max_coords(std::vector<VotMapMaxCoords>& coords, std::string& fname);
bool load_votmax_max_coords(std::string& fname, VotMapMaxCoords& coords);
bool load_votmax_max_coords(std::string& fname, std::vector<VotMapMaxCoords>& coords);
bool save_class_wise_votmaps(std::vector<std::vector<std::vector<cv::Mat> > >& votmaps, std::string& fname);
bool load_class_wise_votmaps(std::string& fname, std::vector<std::vector<std::vector<cv::Mat> > >& votmaps);

#endif // VOTMAPMAXIMA_H
} // namespace learning
} // namespace forest
