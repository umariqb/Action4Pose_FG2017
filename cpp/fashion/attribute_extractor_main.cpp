/*
 * attribute_extractor_main.cpp
 *
 *  Created on: Feb 17, 2012
 *      Author: lbossard
 */


#include <vector>
#include <tr1/unordered_map>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem/path.hpp>
namespace fs = boost::filesystem;

#include "cpp/utils/libsvm/liblinear.hpp"
#include "cpp/utils/file_utils.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/vision/features/bow_histogram.hpp"
namespace vf = vision::features;
namespace ll = utils::liblinear;

#include "segmentation/SegmentedObject.h"

struct FeatureHolder
{
    vf::BowHistogram surf;
    vf::BowHistogram hog;
    vf::BowHistogram color;
    vf::BowHistogram lbp;
    vf::BowHistogram ssd;


    bool has_all() const
    {
        return surf.histogram().data
                && hog.histogram().data
                && color.histogram().data
                && lbp.histogram().data
                && ssd.histogram().data;
    }



    vf::BowHistogram& get(vf::feature_type::T type)
    {
        switch(type)
        {
        case vf::feature_type::Surf:
            return surf;
            break;
        case vf::feature_type::Hog:
            return hog;
            break;
        case vf::feature_type::Color:
            return color;
            break;
        case vf::feature_type::Lbp:
            return lbp;
            break;
        case vf::feature_type::Ssd:
            return ssd;
            break;
        case vf::feature_type::None:
        case vf::feature_type::ColorSsd:
        case vf::feature_type::Gist:
        case vf::feature_type::ColorGist:
        case vf::feature_type::RootSurf:
        case vf::feature_type::FelzenHog:
            std::cerr << "not supported feature type" << std::endl;
            break;
        }
        std::exit(1);
    }



    int feature_dim(int feature_types) const
    {
        int histo_dim = 0;
        if (feature_types & vf::feature_type::Surf)
        {
            histo_dim += surf.histogram().cols;
        }
        if (feature_types & vf::feature_type::Hog)
        {
             histo_dim += hog.histogram().cols;
        }
        if (feature_types & vf::feature_type::Color)
        {
            histo_dim += color.histogram().cols;
        }
        if (feature_types & vf::feature_type::Lbp)
        {
            histo_dim += lbp.histogram().cols;
        }
        if (feature_types & vf::feature_type::Ssd)
        {
            histo_dim += ssd.histogram().cols;
        }
        return histo_dim;
    }

    void assemble_histogram(int feature_types,  std::vector<ll::FeatureNode>& bow_histo) const
    {
        bow_histo.clear();
        bow_histo.reserve(feature_dim(feature_types) + 1);

        int offset = 0;
        if (feature_types & vf::feature_type::Surf)
        {
            offset += copy_histogram(surf, offset, bow_histo);
        }
        if (feature_types & vf::feature_type::Hog)
        {
            offset += copy_histogram(hog, offset, bow_histo);
        }
        if (feature_types & vf::feature_type::Color)
        {
            offset += copy_histogram(color, offset, bow_histo);
        }
        if (feature_types & vf::feature_type::Lbp)
        {
            offset += copy_histogram(lbp, offset, bow_histo);
        }
        if (feature_types & vf::feature_type::Ssd)
        {
            offset += copy_histogram(ssd, offset, bow_histo);
        }
        // add last element
        ll::FeatureNode node;
        node.index = -1;
        node.value = 0;
        bow_histo.push_back(node);
    }

    static int copy_histogram(const vf::BowHistogram& bow_histo, const int offset, std::vector<ll::FeatureNode>& dest)
    {
        typedef vf::BowHistogram::Histogram::value_type ValueType;
        cv::MatConstIterator_<ValueType> it(bow_histo.histogram().begin());
        const cv::MatConstIterator_<ValueType> end = bow_histo.histogram().end();
        ValueType value = 0;
        ll::FeatureNode node;
        for (int index = offset + 1; it < end; ++it, ++index)
        {
            value = *it;
            if (value != static_cast<ValueType>(0) )
            {
                node.index = index;
                node.value = 1; // Max pooling,fun(value);
                dest.push_back(node);
            }
        }
        return bow_histo.histogram().cols;
    }

};

struct LinearConfig
{
    std::string path;
    int feature_types;

    LinearConfig(const std::string& path, int feature_types)
    : path(path), feature_types(feature_types)
    {}

};

struct LinearModel
{
    int feature_types;
    boost::shared_ptr<utils::liblinear::LinearHolder> linear_model;
    LinearModel(){};
    LinearModel(int feature_types, utils::liblinear::LinearHolder* model)
    : feature_types(feature_types), linear_model(model)
    {};
};

std::vector<LinearConfig> get_config()
{
    std::vector<LinearConfig> c;

    // sleeves
    c.push_back(LinearConfig(
            "/home/lbossard/scratch/datasets/google-fashion/features/attributes/"
            "f=25_p=3_mpool=0/eval/t=clothing_e=clothing/results/"
            "c=1_s=6/sleeves/linear_model",
            25)); // 'lbp', 'surf', 'ssd'
    // color
    c.push_back(LinearConfig(
            "/home/lbossard/scratch/datasets/google-fashion/features/attributes/"
            "f=28_p=3_mpool=0/eval/t=clothing_e=clothing/results/"
            "c=1_s=2/color/linear_model",
            28)); //'color', 'lbp', 'ssd'
    // general
    c.push_back(LinearConfig(
            "/home/lbossard/scratch/datasets/google-fashion/features/attributes/"
            "f=20_p=3_mpool=0/eval/t=clothing_e=clothing/results/"
            "c=1_s=2/general/linear_model",
            20)); //'color', 'ssd'
    // person
    c.push_back(LinearConfig(
            "/home/lbossard/scratch/datasets/google-fashion/features/attributes/"
            "f=23_p=3_mpool=0/eval/t=clothing_e=clothing/results/"
            "c=1_s=2/person/linear_model",
            23)); //'color', 'ssd'
    // material
    c.push_back(LinearConfig(
            "/home/lbossard/scratch/datasets/google-fashion/features/attributes/"
            "f=31_p=3_mpool=0/eval/t=clothing_e=clothing/results/"
            "c=1_s=6/material/linear_model",
            31)); //'lbp', 'surf', 'hog', 'color', 'ssd'
    // structure
    c.push_back(LinearConfig(
            "/home/lbossard/scratch/datasets/google-fashion/features/attributes/"
            "f=16_p=3_mpool=0/eval/t=clothing_e=clothing/results/"
            "c=1_s=6/structure/linear_model",
            16)); // 'ssd'
    // style
    c.push_back(LinearConfig(
            "/home/lbossard/scratch/datasets/google-fashion/features/attributes/"
            "f=16_p=3_mpool=0/eval/t=clothing_e=clothing/results/"
            "c=1_s=2/style/linear_model",
            16)); //'ssd'
    // pattern
    c.push_back(LinearConfig(
            "/home/lbossard/scratch/datasets/google-fashion/features/attributes/"
            "f=16_p=3_mpool=0/eval/t=clothing_e=clothing/results/"
            "c=1_s=6/pattern/linear_model",
            16)); //'ssd'

    return c;
};

void load_svm_or_die(const std::string& svm_file, utils::liblinear::LinearHolder& svm)
{
    if (!svm.read_from_text(svm_file))
    {
        std::cerr << "could not load svm from: " << svm_file << std::endl;
        std::exit(1);
    }
}

typedef std::vector<LinearModel> ModelCollection;
ModelCollection load_attribute_models()
{
    std::vector<LinearConfig> c = get_config();

    std::vector<LinearModel> models;
    models.reserve(c.size());
    BOOST_FOREACH(const LinearConfig& config, c)
    {
        LinearModel m(config.feature_types, new utils::liblinear::LinearHolder());
        load_svm_or_die(config.path, *m.linear_model);
        models.push_back(m);
    }
    return models;
}

ModelCollection load_class_model()
{
    std::vector<LinearConfig> c = get_config();

    std::vector<LinearModel> models;
    models.reserve(1);
    LinearModel m(15, new utils::liblinear::LinearHolder());
    load_svm_or_die(
            "/usr/biwinas02/scratch-g/lbossard/image-net.org/fashion/svmfeatures/f=15_p=3_mpool=0/eval_extended/s=6_c=1_weighting=quadratic/linear_model",
            *m.linear_model);
    models.push_back(m);
    return models;
}

int get_total_classe_count(const ModelCollection& models)
{
    int class_count = 0;
    BOOST_FOREACH(const LinearModel& m, models)
    {
        class_count += m.linear_model->class_count();
    }
    return class_count;
}

typedef std::tr1::unordered_map<std::string, FeatureHolder > FeatureFileCollection;
void loadFeatures(const std::string feature_dir, vf::feature_type::T feature_type, FeatureFileCollection& fc)
{

    // find all feature files by extension
    typedef std::vector<fs::path> FileCollection;
    FileCollection feature_files;
    utils::fs::collect_files(
            feature_dir,
            ".*\\." + vf::feature_type::to_string(feature_type),
            std::back_inserter(feature_files));

    std::size_t f_count = 0;
    BOOST_FOREACH(const fs::path& p, feature_files)
    {
        std::string obj_name = p.filename().replace_extension("").string();
        vf::BowHistogram& bow_histogram = fc[obj_name].get(feature_type);
        // try to load
        if (utils::serialization::read_binary_archive(p.string(), bow_histogram))
        {
            ++f_count;
        }
        else
        {
            // otherwhise erase
            bow_histogram.histogram().create(0,0);
        }
    }
    std::cout << "lodaded " << f_count << " " << vf::feature_type::to_string(feature_type) << std::endl;
}



DEFINE_string(feature_dir, "", "path to features");
DEFINE_string(output_dir, "", "directory to save the features");

DEFINE_string(model_type, "attributes", "which model to use. attributes or classes");
DEFINE_double(sigmoid_alpha, 0., "fit a sigmoid. 0. == false. otherwhise value is taken as a: 1./(1 + exp(-a * x))");
DEFINE_bool(as_probability, false, "vector sums up to one. NOTE: fit_sigmoid needs to be != 0");
int main(int argc, char **argv)
{
     // init logging
    google::InitGoogleLogging(argv[0]);

    // get command line args
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_feature_dir.empty()
                    ||  FLAGS_output_dir.empty()
                    || (FLAGS_model_type != "attributes" && FLAGS_model_type != "classes")
                    )
    {
        google::ShowUsageWithFlags(argv[0]);
        return 1;
    }

    const std::string feature_dir     = FLAGS_feature_dir;
    const fs::path output_dir         = FLAGS_output_dir;
    const std::string model_type      = FLAGS_model_type;
    const bool fit_sigmoid            = (FLAGS_sigmoid_alpha != 0.);
    const double sigmoid_alpha	      = FLAGS_sigmoid_alpha;
    const bool convert_to_probability = fit_sigmoid && FLAGS_as_probability;


    // load  all bow histograms to the memroy
    typedef std::tr1::unordered_map<std::string, FeatureHolder > FeatureFileCollection;
    FeatureFileCollection fc;
    loadFeatures(feature_dir, vf::feature_type::Surf, fc);
    loadFeatures(feature_dir, vf::feature_type::Hog, fc);
    loadFeatures(feature_dir, vf::feature_type::Color, fc);
    loadFeatures(feature_dir, vf::feature_type::Lbp, fc);
    loadFeatures(feature_dir, vf::feature_type::Ssd, fc);


    // load all svms
    ModelCollection model_collection;
    std::string suffix;
    if (model_type == "attributes")
    {
        model_collection = load_attribute_models();
        suffix = ".attrs";
    }
    else if (model_type == "classes")
    {
        model_collection = load_class_model();
        suffix = ".classs";
    }
    const int class_count = get_total_classe_count(model_collection);
//    BOOST_FOREACH(const LinearModel& model, model_collection)
//    {
//        for (int i = 0; i < model.linear_model->get_model().nr_class; ++i)
//        {
//            std::cout << i << ":" << model.linear_model->get_model().label[i] << ", ";
//        }
//        std::cout << std::endl;
//    }

    // go through each file
    std::cout << "processing " << fc.size() << " objects" << std::endl;
    boost::progress_display progress(fc.size());
    int sucess_files = 0;
    BOOST_FOREACH( const FeatureFileCollection::value_type& fh_pair, fc)
    {
        ++progress;
        const FeatureHolder& feature_holder = fh_pair.second;
        const std::string file_path = fh_pair.first;
        if (!feature_holder.has_all())
        {
            continue;
        }


        // get the attribute values for each model
        cv::Mat_<double> histogram(1, class_count);
        std::vector<ll::FeatureNode> liblinear_features;
        std::vector<double> values;
        int offset = 0;
        BOOST_FOREACH(const LinearModel& model, model_collection)
        {
            // assemble feature vector
            feature_holder.assemble_histogram(model.feature_types, liblinear_features);

            // convert

            // classify
            int label = model.linear_model->predict(liblinear_features, values);
//std::cout << label << std::endl;
//            std::cout << cv::Mat_<double>(values) << std::endl;

            // put distances to cv mat
            CHECK(offset + values.size() <= histogram.cols);
            for (int i = 0; i < values.size(); ++i, ++offset)
            {
                 histogram(offset) = values[i];
            }
        }

        // fit sigmoid if necessary
        if (fit_sigmoid)
        {
            histogram *= -sigmoid_alpha;
            cv::exp(histogram, histogram);
            histogram = 1. / (1. + histogram);
        }

        // probability
        if (convert_to_probability)
        {
            histogram /= cv::sum(histogram)[0];
        }

        // save vector for this segmented object
        std::string histo_path = (output_dir / fs::path(file_path).filename()).string() + suffix;
        if (!utils::serialization::write_binary_archive(histo_path, histogram))
        {
            std::cerr << "could not write to " << histo_path << std::endl;
            return 1;
        }

        ++sucess_files;
    }
    std::cout << "processed " << sucess_files << " features" << std::endl;
    return 0;
}

