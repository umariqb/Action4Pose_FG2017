#include <istream>

#include <opencv2/opencv.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "cpp/utils/image_file_utils.hpp"

DEFINE_string(index_file, "", "Index of test images");
DEFINE_string(other_index_file, "", "Another index");

int main(int argc, char** argv) {
	google::ParseCommandLineFlags(&argc, &argv, true);
	CHECK(!FLAGS_index_file.empty()) << "No index file provided.";

	std::vector<std::string> index_files;
	index_files.push_back(FLAGS_index_file);
	if (!FLAGS_other_index_file.empty()) {
		index_files.push_back(FLAGS_other_index_file);
	}
	utils::image_file::ImagePatchIterator patch_it(index_files,
																								 cv::Size(100, 80),
																								 3, 0.5, 50);
	while (true) {
		cv::Mat p = patch_it.next();
		cv::imshow("gigi", p);
		cv::waitKey(0);
	}
	return 0;
}
