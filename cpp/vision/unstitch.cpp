#include "unstitch.hpp"

#include <glog/logging.h>


namespace vision
{
    Unstitch::Unstitch()
    {
        this->min_score_for_split_ = 25.0f;
        this->min_part_fraction_required_ = 0.2f;
        this->fraction_of_pixels_for_edge_ = 0.7f;
        this->fraction_of_pixels_cropped_from_edge_ = 0.1f;
        this->fraction_for_close_edge_ = 0.6f;
    }

    template<typename T>
    static inline T square(T value)
    {
        return value * value;
    }

    void Unstitch::unstitchImage(const cv::Mat& image, std::vector<cv::Rect>& parts)
    {
        cv::Mat horizontalEdges(image.rows, image.cols, CV_8UC1);
        cv::Mat verticalEdges(image.cols, image.rows, CV_8UC1);

        for (int r = 0; r < image.rows; ++r)
        {
            horizontalEdges.data[r * horizontalEdges.step[0]] = 0.0f;
            verticalEdges.data[r * verticalEdges.step[1]] = 0.0f;
        }

        for (int c = 0; c < image.cols; ++c)
        {
            horizontalEdges.data[c * horizontalEdges.step[1]] = 0.0f;
            verticalEdges.data[c * verticalEdges.step[0]] = 0.0f;
        }

        for (int r = 1; r < image.rows; ++r)
        {
            for (int c = 1; c < image.cols; ++c)
            {
                unsigned char* r0c0 = image.data + (r - 1) * image.step[0] + (c - 1) * image.step[1];
                unsigned char* r0c1 = image.data + (r - 1) * image.step[0] + c * image.step[1];
                unsigned char* r1c0 = image.data + r * image.step[0] + (c - 1) * image.step[1];
                unsigned char* r1c1 = image.data + r * image.step[0] + c * image.step[1];

                float diff_r = 0;
                float diff_c = 0;

                for (int channel = 0; channel < image.channels(); ++channel)
                {
                    diff_r += square(r0c0[channel] - r1c0[channel]);
                    diff_r += square(r0c1[channel] - r1c1[channel]);

                    diff_c += square(r0c0[channel] - r0c1[channel]);
                    diff_c += square(r1c0[channel] - r1c1[channel]);
                }

                static float factor = 255.0f / sqrt(3*255*255);
                diff_r = sqrt(diff_r) * factor;
                diff_c = sqrt(diff_c) * factor;

                horizontalEdges.data[r * horizontalEdges.step[0] + c * horizontalEdges.step[1]] = std::max(0.0f, diff_r - diff_c);
                verticalEdges.data[c * verticalEdges.step[0] + r * verticalEdges.step[1]] = std::max(0.0f, diff_c - diff_r);
            }
        }

        int stop = this->min_part_fraction_required_ * std::min(image.rows, image.cols);
        this->unstitchRecursive(image, stop, horizontalEdges, verticalEdges, parts);
    }

    void Unstitch::unstitchRecursive(const cv::Mat& image, int stop, const cv::Mat& horizontalEdges, const cv::Mat& verticalEdges, std::vector<cv::Rect>& parts)
    {
        // both horizontalEdges and verticalEdges have the edges stored horizontally since verticalEdges is rotated by 90 degree

        // get the stronges edges in both directions
        SplitData horizontalSplit = this->unstitchRecursive1D(stop, horizontalEdges);
        SplitData verticalSplit = this->unstitchRecursive1D(stop, verticalEdges);

        LOG(INFO) << "horizontal: " << horizontalSplit.score << " at index " << horizontalSplit.index << std::endl;
        LOG(INFO) << "vertical: " << verticalSplit.score << " at index " << verticalSplit.index << std::endl;

        // check if at least one edge is strong enough to perform a split
        std::vector<int> horizontalSplits;
        std::vector<int> verticalSplits;
        if (horizontalSplit.score > this->min_score_for_split_ || verticalSplit.score > this->min_score_for_split_)
        {
            // find the strongest edge overall and prepare the vectors. each vector defines the splits in one direction with 0 and rows/cols also being "splits"
            horizontalSplits.clear();
            verticalSplits.clear();

            horizontalSplits.push_back(0);
            verticalSplits.push_back(0);

            if (horizontalSplit.score > verticalSplit.score)
            {
                horizontalSplits.push_back(horizontalSplit.index);
                LOG(INFO) << "=> split horizontal at " << horizontalSplit.index << std::endl;
            }
            else
            {
                verticalSplits.push_back(verticalSplit.index);
                LOG(INFO) << "=> split vertical at " << verticalSplit.index << std::endl;
            }

            horizontalSplits.push_back(image.rows);
            verticalSplits.push_back(image.cols);

            bool bHasSubparts = false;
            for (size_t r = 0; r < horizontalSplits.size() - 1; ++r)
            {
                int height = horizontalSplits[r + 1] - horizontalSplits[r];
                if (height < stop)
                {
                    continue;
                }

                for (size_t c = 0; c < verticalSplits.size() - 1; ++c)
                {
                    int width = verticalSplits[c + 1] - verticalSplits[c];
                    if (width < stop)
                    {
                        continue;
                    }

                    bHasSubparts = true;

                    cv::Rect roi_horizontal(verticalSplits[c], horizontalSplits[r], width, height);
                    cv::Rect roi_vertical(horizontalSplits[r], verticalSplits[c], height, width);

                    this->unstitchRecursive(image(roi_horizontal), stop, horizontalEdges(roi_horizontal), verticalEdges(roi_vertical), parts);
                }
            }

            if (bHasSubparts)
            {
                return;
            }
        }

        LOG(INFO) << "=> don't split" << std::endl;

        cv::Point offset;
        cv::Size size;
        image.locateROI(size, offset);

        parts.push_back(cv::Rect(offset, image.size()));
    }

    Unstitch::SplitData Unstitch::unstitchRecursive1D(int stop, const cv::Mat& horizontalEdges)
    {
        // for each row we calculate the mean of the edge values. we start at row 1 since row 0 may not contain an edge
        std::vector<float> means(horizontalEdges.rows + 1, 0);
        means[0] = means[horizontalEdges.rows] = 0; // initialize end points with zero

        for (int r = 1; r < horizontalEdges.rows; ++r)
        {
            means[r] = 0;
            for (int c = 0; c < horizontalEdges.cols; ++c)
            {
                means[r] += horizontalEdges.data[r * horizontalEdges.step[0] + c * horizontalEdges.step[1]];
            }
            means[r] /= horizontalEdges.cols;
        }

        // highlight single edges and calculate the average mean
        float average_mean = 0;
        for (int r = 1; r < horizontalEdges.rows; ++r)
        {
            means[r] = std::max(0.0, 2.0 * means[r] - 1.0 * (means[r - 1] + means[r + 1]));
            average_mean += means[r];
        }
        average_mean /= (horizontalEdges.rows - 1);

        // for each row find a value such that 50% of the edge intensities are above this value
        std::vector<float> thresholds(horizontalEdges.rows);
        thresholds[0] = 0;

        for (int r = 1; r < horizontalEdges.rows; ++r)
        {
            // create a histogram of the edge intensities
            float histogram[256] = {0}; // initialize all with zero
            for (int c = 0; c < horizontalEdges.cols; ++c)
            {
                unsigned char value = horizontalEdges.data[r * horizontalEdges.step[0] + c * horizontalEdges.step[1]];
                ++histogram[value];
            }

            // find the 50% threshold
            thresholds[r] = 255;
            int sum = 0;
            for (size_t h = 0; h < 256; ++h)
            {
                sum += h;
                if (sum > (1.0f - this->fraction_of_pixels_for_edge_) * horizontalEdges.cols)
                {
                    thresholds[r] = h;
                    break;
                }
            }
        }

        // for each row compute the length of the edge
        std::vector<float> lengths(horizontalEdges.rows);
        lengths[0] = 0;

        for (int r = 1; r < horizontalEdges.rows; ++r)
        {
            int count;
            int c_left;
            int c_right;

            // cut off the leftmost 10% of pixels above the threshold
            for (c_left = 0, count = 0; c_left < horizontalEdges.cols && count < this->fraction_of_pixels_cropped_from_edge_ * horizontalEdges.cols; ++c_left){
                if (horizontalEdges.data[r * horizontalEdges.step[0] + c_left * horizontalEdges.step[1]] > thresholds[r])
                {
                    count++;
                }
            }

            // cut off the rightmost 10% of pixels above the threshold
            for (c_right = horizontalEdges.cols - 1, count = 0; c_right >= 0 && count < this->fraction_of_pixels_cropped_from_edge_ * horizontalEdges.cols; --c_right)
            {
                if (horizontalEdges.data[r * horizontalEdges.step[0] + c_right * horizontalEdges.step[1]] > thresholds[r])
                {
                    count++;
                }
            }

            // the length is the difference of the right and the left endpoint
            lengths[r] = std::max(0, c_right - c_left);
            lengths[r] /= horizontalEdges.cols;
        }

        // calculate the score for each row, based on strength and average strength
        std::vector<float> scores(horizontalEdges.rows + 1);
        scores[0] = scores[horizontalEdges.rows] = 0;

        for (int r = 1; r < horizontalEdges.rows; ++r)
        {
            scores[r] = means[r] * lengths[r];
        }

        // filter out weak edges which are next to a stronger edge
        for (int r = 1; r < horizontalEdges.rows; ++r)
        {
            if (scores[r] == 0)
            {
                continue;
            }

            // remove weaker edges to the left
            float temp = scores[r];
            for (int rr = r - 1; rr > 0; --rr)
            {
                if (scores[rr] < temp)
                {
                    temp = scores[rr];
                    scores[rr] = 0;
                }
            }

            // remove weaker edges to the right
            temp = scores[r];
            for (int rr = r + 1; rr < horizontalEdges.rows; ++rr)
            {
                if (scores[rr] < temp)
                {
                    temp = scores[rr];
                    scores[rr] = 0;
                }
            }
        }

        // filter out strong edges which are close to 2 or more other strong edges
        std::vector<float> filtered_scores(horizontalEdges.rows);
        filtered_scores[0] = 0;

        for (int r = 1; r < horizontalEdges.rows; ++r)
        {
            float threshold = this->fraction_for_close_edge_ * scores[r];

            int count = 0;
            for (int r_offset = 1; r_offset < stop / 2; ++r_offset)
            {
                // check to the left
                if (r - r_offset > 0 && scores[r - r_offset] > threshold)
                {
                    count++;
                }

                // check to the right
                if (r + r_offset < horizontalEdges.rows && scores[r + r_offset] > threshold)
                {
                    count++;
                }
            }

            if (count > 1)
            {
                filtered_scores[r] = 0;
            }
            else
            {
                filtered_scores[r] = scores[r];
            }
        }

        // find the edge with the highest score after filtering
        int max_index = 0;
        float max_score = 0;

        for (int r = 1; r < horizontalEdges.rows; ++r)
        {
            if (filtered_scores[r] > max_score)
            {
                max_score = filtered_scores[r];
                max_index = r;
            }
        }

        SplitData split;
        split.index = max_index;
        split.score = max_score;

        return split;
    }
}
