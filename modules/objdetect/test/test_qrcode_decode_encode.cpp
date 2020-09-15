// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
namespace opencv_test { namespace {

std::string qrcode_images_name[] = {
        "v2_c0_mask2_mode7_eci26.png",
        "v1_c0_mask0_mode1_eci26.png",  "v2_c0_mask7_mode4_eci26.png",
        "v1_c0_mask0_mode2_eci26.png",  "v4_c0_mask2_mode4_eci26.png",
        "v1_c0_mask7_mode4_eci26.png",  "v5_c0_mask2_mode4_eci26.png",
        "v1_c2_mask3_mode2_eci26.png"
};

const Size fixed_size = Size(600,600);
const int border_width = 2;
const int auto_mode = -1;

//#define UPDATE_TEST_DATA
//#define  UPDATE_DECODE_TEST_DATA

#ifdef UPDATE_TEST_DATA
        TEST(Objdetect_QRCode, generate_test_data)
        {
            const std::string root = "qrcode/decode_encode";
            const std::string dataset_config = findDataFile(root +"/"+ "dataset_config.json");
            FileStorage file_config(dataset_config, FileStorage::WRITE);

            file_config << "test_images" << "[";
            size_t images_count = sizeof(qrcode_images_name) / sizeof(qrcode_images_name[0]);
            for (size_t i = 0; i < images_count; i++)
            {
                file_config << "{:" << "image_name" << qrcode_images_name[i];
                std::string image_path = findDataFile(root +"/"+ qrcode_images_name[i]);

                /**read from test set*/
                Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
                src = src(Range(border_width,src.rows-border_width),Range(border_width,src.rows-border_width)).clone();
//                std::cout<<image_path<<":"<<src.rows<<'\n';
//                cout<<src<<endl;

                resize(src,src,fixed_size,0,0,INTER_AREA);

//                for(int i = 0 ; i < src.rows ; i++ ){
//                    std::cout<<(int)src.ptr(0)[i]<<" ";
//                }
//                std::cout<<"\n";

                std::string decoded_info;
                ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

                /**add the corner points*/
                std::vector<Point> corners(4);

                corners[0] = Point(0,0);
                corners[1] = Point(src.rows-1,0);
                corners[2] = Point(src.rows-1,src.cols-1);
                corners[3] = Point(0,src.cols-1);

                EXPECT_TRUE(decodeQRCode(src, corners, decoded_info, straight_barcode))<< "ERROR : " << image_path;

                file_config << "x" << "[:";
                for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
                file_config << "]";
                file_config << "y" << "[:";
                for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
                file_config << "]";
                /**use escape character for alternative interpretation in a character's sequence for “]” */
                for(size_t t = 0 ; t < decoded_info.size() ; t++ ){
                    if(decoded_info[t] == ']'||decoded_info[t] == '}'||decoded_info[t] == '['||decoded_info[t] == '{'){
                        decoded_info.insert(t,"\\");
                        t+=2;
                    }
                }

                file_config << "info" << decoded_info;

                file_config << "}";
            }
            file_config << "]";
            file_config.release();
        }
#else

//#ifdef UPDATE_ENCODE_TEST_DATA
//        TEST(Objdetect_QRCode, generate_test_data)
//        {
//            const std::string root = "qrcode/encode";
//            const std::string dataset_config = findDataFile(root +"/"+ "dataset_config.json");
//            FileStorage file_config(dataset_config, FileStorage::WRITE);
//
//            file_config << "test_images" << "[";
//            size_t images_count = sizeof(qrcode_images_name) / sizeof(qrcode_images_name[0]);
//            for (size_t i = 0; i < images_count; i++)
//            {
//                file_config << "{:" << "image_name" << qrcode_images_name[i];
//                std::string image_path = findDataFile(root +"/"+ qrcode_images_name[i]);
//                std::vector<Point> corners;
//                Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
//                std::string decoded_info;
//                ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
//
//                /**add the corner points*/
//                corners.push_back(Point(0,0));
//                corners.push_back(Point(src.rows-1,0));
//                corners.push_back(Point(src.rows-1,src.cols-1));
//                corners.push_back(Point(0,src.cols-1));
//
//                int mode;
//                int version;
//                int ecc_level;
//                int mask_type;
//                int eci_num;
//
//                EXPECT_TRUE(decodeQRCode(src, corners, decoded_info, straight_barcode,mode,version,ecc_level,mask_type,eci_num));
//
//                file_config << "x" << "[:";
//                for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
//                file_config << "]";
//                file_config << "y" << "[:";
//                for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
//                file_config << "]";
//                file_config << "info" << decoded_info;
//                file_config << "version" << version;
//                file_config << "mode" << mode;
//                file_config << "ecc_level" << ecc_level;
//                file_config << "mask_type" << mask_type;
//
//                file_config << "}";
//            }
//            file_config << "]";
//            file_config.release();
//        }
//#endif

//#if (!defined UPDATE_DECODE_TEST_DATA) && (!defined UPDATE_ENCODE_TEST_DATA)
typedef testing::TestWithParam< std::string > Objdetect_QRCode_Decode;
TEST_P(Objdetect_QRCode_Decode, regression)
{
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/decode_encode";

    std::string image_path = findDataFile(root +"/"+ name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    src = src(Range(border_width,src.rows-border_width),Range(border_width,src.rows-border_width)).clone();
    resize(src,src,fixed_size);

    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    const std::string dataset_config = findDataFile(root + "/"+"dataset_config.json");
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
                std::vector<Point> corners;
                std::string decoded_info;

                for (int i = 0; i < 4; i++)
                {
                    int x = config["x"][i];
                    int y = config["y"][i];
                    corners.push_back(Point(x,y));
                }

                EXPECT_TRUE(decodeQRCode(src, corners, decoded_info, straight_barcode));
                ASSERT_FALSE(decoded_info.empty());

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
            const std::string name_current_image = GetParam();
            const std::string root = "qrcode/decode_encode";

            std::string image_path = findDataFile(root +"/"+ name_current_image);
            /**read the original image */
            Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
            ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
            src = src(Range(border_width,src.rows-border_width),Range(border_width,src.rows-border_width)).clone();
            resize(src,src,fixed_size);
            const std::string dataset_config = findDataFile(root +"/"+ "dataset_config.json");
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
                        QRCodeEncoder encoder;
                        Mat result = encoder.generateSingle(original_info,auto_mode);
                        resize(result,result,fixed_size,0,0,INTER_AREA);
                        std::vector<Point> corners;
                        std::string decoded_info;

                        for (int i = 0; i < 4; i++)
                        {
                            int x = config["x"][i];
                            int y = config["y"][i];
                            corners.push_back(Point(x,y));
                        }
                        /**the encoded info should be the same as the original info*/
                        bool success = decodeQRCode(result, corners, decoded_info, straight_barcode);
                        ASSERT_TRUE(success&&(original_info == decoded_info)) << "The generated QRcode is not same as test data:" << name_test_image;

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

INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Decode, testing::ValuesIn(qrcode_images_name));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Encode, testing::ValuesIn(qrcode_images_name));

#endif // UPDATE_QRCODE_TEST_DATA

}} // namespace
