#ifndef VISION_UNSTITCH_H__
#define VISION_UNSTITCH_H__

#include <vector>
#include <opencv2/core/core.hpp>

namespace vision
{
    class Unstitch
    {
        struct SplitData
        {
            int index;
            float score;
        };

        public:
            Unstitch();

            void unstitchImage(
                    const cv::Mat& image,
                    std::vector<cv::Rect>& parts);

            float min_score_for_split_; // edge-score which is required to split an image
            float min_part_fraction_required_; // how much smaller than the image a part can be
            float fraction_of_pixels_for_edge_; // how many pixels are assumed to form an edge in % of the image size
            float fraction_of_pixels_cropped_from_edge_; // how many pixels are cropped from an edge to find the end-points of an edge in % of the image size
            float fraction_for_close_edge_; // how much weaker an edge can be to be still considered "close"

        private:
            void unstitchRecursive(
                    const cv::Mat& image,
                    int stop,
                    const cv::Mat& horizontalEdges,
                    const cv::Mat& verticalEdges,
                    std::vector<cv::Rect>& parts);

            SplitData unstitchRecursive1D(
                    int stop,
                    const cv::Mat& horizontalEdge);
    };
}

#endif /* _UNSTITCH_H__ */
