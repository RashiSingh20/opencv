// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

std::string qrcode_images_name[] = {
        "v2_c0_mask7_mode4_eci26.png", "v1_c0_mask7_mode4_eci26.png", "v4_c0_mask2_mode4_eci26.png" ,"v2_c0_mask2_mode7_eci26.png",
        "v1_c0_mask0_mode1_eci26.png", "v1_c2_mask3_mode2_eci26.png", "v1_c2_mask0_mode2_eci26.png"
};

//#define UPDATE_ENCODE_TEST_DATA
//#define  UPDATE_DECODE_TEST_DATA

#ifdef UPDATE_DECODE_TEST_DATA
        TEST(Objdetect_QRCode, generate_test_data)
        {
            const std::string root = "qrcode/decode";
            const std::string dataset_config = findDataFile(root + "dataset_config.json");
            FileStorage file_config(dataset_config, FileStorage::WRITE);

            file_config << "test_images" << "[";
            size_t images_count = sizeof(qrcode_images_name) / sizeof(qrcode_images_name[0]);
            for (size_t i = 0; i < images_count; i++)
            {
                file_config << "{:" << "image_name" << qrcode_images_name[i];
                std::string image_path = findDataFile(root + qrcode_images_name[i]);
                std::vector<Point> corners;
                Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
                std::string decoded_info;
                ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

                /**add the corner points*/
                corners.push_back(Point(0,0));
                corners.push_back(Point(src.rows-1,0));
                corners.push_back(Point(src.rows-1,src.cols-1));
                corners.push_back(Point(0,src.cols-1));

                EXPECT_TRUE(decodeQRCode(src, corners, decoded_info, straight_barcode));

                file_config << "x" << "[:";
                for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
                file_config << "]";
                file_config << "y" << "[:";
                for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
                file_config << "]";
                file_config << "info" << decoded_info;

                file_config << "}";
            }
            file_config << "]";
            file_config.release();
        }
#endif

#ifdef UPDATE_ENCODE_TEST_DATA
        TEST(Objdetect_QRCode, generate_test_data)
        {
            const std::string root = "qrcode/encode";
            const std::string dataset_config = findDataFile(root + "dataset_config.json");
            FileStorage file_config(dataset_config, FileStorage::WRITE);

            file_config << "test_images" << "[";
            size_t images_count = sizeof(qrcode_images_name) / sizeof(qrcode_images_name[0]);
            for (size_t i = 0; i < images_count; i++)
            {
                file_config << "{:" << "image_name" << qrcode_images_name[i];
                std::string image_path = findDataFile(root + qrcode_images_name[i]);
                std::vector<Point> corners;
                Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
                std::string decoded_info;
                ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

                /**add the corner points*/
                corners.push_back(Point(0,0));
                corners.push_back(Point(src.rows-1,0));
                corners.push_back(Point(src.rows-1,src.cols-1));
                corners.push_back(Point(0,src.cols-1));

                int mode;
                int version;
                int ecc_level;
                int mask_type;
                int eci_num;

                EXPECT_TRUE(decodeQRCode(src, corners, decoded_info, straight_barcode,mode,version,ecc_level,mask_type,eci_num));

                file_config << "x" << "[:";
                for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
                file_config << "]";
                file_config << "y" << "[:";
                for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
                file_config << "]";
                file_config << "info" << decoded_info;
                file_config << "version" << version;
                file_config << "mode" << mode;
                file_config << "ecc_level" << ecc_level;
                file_config << "mask_type" << mask_type;

                file_config << "}";
            }
            file_config << "]";
            file_config.release();
        }
#endif


#if (!defined UPDATE_DECODE_TEST_DATA) && (!defined UPDATE_ENCODE_TEST_DATA)
typedef testing::TestWithParam< std::string > Objdetect_QRCode_Decode;
TEST_P(Objdetect_QRCode_Decode, regression)
{
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/decode";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;


    std::vector<Point> corners;
    std::string decoded_info;
    QRCodeDetector qrcode;

    corners.push_back(Point(0,0));
    corners.push_back(Point(src.rows-1,0));
    corners.push_back(Point(src.rows-1,src.cols-1));
    corners.push_back(Point(0,src.cols-1));

    decoded_info = qrcode.detectAndDecode(src, corners, straight_barcode);
    EXPECT_TRUE(decodeQRCode(src, corners, decoded_info, straight_barcode));
    ASSERT_FALSE(decoded_info.empty());

    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    {
        FileNode images_list = file_config["test_images"];
        size_t images_count = static_cast<size_t>(images_list.size());
        ASSERT_GT(images_count, 0u) << "Can't find validation data entries in 'test_images': " << dataset_config;

        for (size_t index = 0; index < images_count; index++)
        {
            FileNode config = images_list[(int)index];
            std::string name_test_image = config["image_name"];
            if (name_test_image == name_current_image)
            {

                std::string original_info = config["info"];
                EXPECT_EQ(decoded_info, original_info);

                return; // done
            }
        }
        std::cerr
            << "Not found results for '" << name_current_image
            << "' image in config file:" << dataset_config << std::endl
            << "Re-run tests with enabled UPDATE_DECODE_TEST_DATA macro to update test data."
            << std::endl;
    }
}

        typedef testing::TestWithParam< std::string > Objdetect_QRCode_Encode;
        TEST_P(Objdetect_QRCode_Encode, regression){
            const double ecc_capacity[4] = {0.07,0.15,0.25,0.30};

            const std::string name_current_image = GetParam();
            const std::string root = "qrcode/encode";

            std::string image_path = findDataFile(root + name_current_image);
            /**read the original image */
            Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
            ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

            std::vector<Point> corners;
            std::string decoded_info;

            const std::string dataset_config = findDataFile(root + "dataset_config.json");
            FileStorage file_config(dataset_config, FileStorage::READ);
            ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
            {
                FileNode images_list = file_config["test_images"];
                size_t images_count = static_cast<size_t>(images_list.size());
                ASSERT_GT(images_count, 0u) << "Can't find validation data entries in 'test_images': " << dataset_config;

                for (size_t index = 0; index < images_count; index++)
                {
                    FileNode config = images_list[(int)index];
                    std::string name_test_image = config["image_name"];

                    if (name_test_image == name_current_image)
                    {
                        int mode = config["mode"];
                        int version = config["version"];; 
                        int ecc_level = config["ecc_level"];;
                        int mask_type = config["mask_type"];;
                        int eci_num = config["eci_num"];;
                        std::string original_info = config["info"];
                        QRCodeEncoder encoder;
                        Mat result = encoder.generate(original_info,mode,version ,ecc_level,mask_type,eci_num,1)[0];

                        cv::Mat diff = result != src;
                        int min = result.rows*result.cols;
                        bool eq = cv::countNonZero(diff) < int(min * ecc_capacity[ecc_level]);
                        ASSERT_TRUE(eq) << "The generated QRcode is not same as test data:" << name_test_image;

                        return; // done
                    }
                }
                std::cerr
                        << "Not found results for '" << name_current_image
                        << "' image in config file:" << dataset_config << std::endl
                        << "Re-run tests with enabled UPDATE_ENCODE_TEST_DATA macro to update test data."
                        << std::endl;
            }
        }


#endif // UPDATE_QRCODE_TEST_DATA

}} // namespace
