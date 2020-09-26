// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
namespace opencv_test { namespace {
        std::string decode_qrcode_images_name[] = {
                "v2_c0_mask2_mode7_eci26.png",
                "v1_c0_mask0_mode1_eci26.png",  "v2_c0_mask4_mode4_eci26.png",
                "v1_c0_mask0_mode2_eci26.png",  "v2_c0_mask7_mode4_eci26.png",
                "v1_c0_mask2_mode3_eci26.png",  "v3_c1_mask1_mode5_eci26.png",
                "v1_c0_mask5_mode3_eci26.png",  "v3_c1_mask3_mode3_eci26.png",
                "v1_c0_mask7_mode4_eci26.png",  "v3_c1_mask3_mode9_eci26.png",
                "v1_c2_mask0_mode2_eci26.png",  "v3_c1_mask5_mode5_eci26.png",
                "v1_c2_mask3_mode2_eci26.png",  "v4_c0_mask2_mode4_eci26.png",
                "v2_c0_mask2_mode5_eci26.png",  "v5_c0_mask2_mode4_eci26.png"
        };
        std::string encode_qrcode_images_name[] = {
                "v2_c0_mask2_mode7_eci26.png",
                "v1_c0_mask0_mode1_eci26.png",  "v2_c0_mask4_mode4_eci26.png",
                "v1_c0_mask0_mode2_eci26.png",  "v2_c0_mask7_mode4_eci26.png",
                "v1_c0_mask2_mode3_eci26.png",  "v3_c1_mask1_mode5_eci26.png",
                "v1_c0_mask5_mode3_eci26.png",  "v3_c1_mask3_mode3_eci26.png",
                "v1_c0_mask7_mode4_eci26.png",  "v3_c1_mask3_mode9_eci26.png",
                "v1_c2_mask0_mode2_eci26.png",  "v3_c1_mask5_mode5_eci26.png",
                "v1_c2_mask3_mode2_eci26.png",  "v4_c0_mask2_mode4_eci26.png",
                "v2_c0_mask2_mode5_eci26.png",  "v5_c0_mask2_mode4_eci26.png"
        };
        std::string encode_decode_qrcode_images_name[] = {" "};

        const Size fixed_size = Size(600,600);
        const int border_width = 2;
        const int auto_mode = -1;
        int countDiffPixels(cv::Mat in1, cv::Mat in2);
        int countDiffPixels(cv::Mat in1, cv::Mat in2) {
            cv::Mat diff;
            cv::compare(in1, in2, diff, cv::CMP_NE);
            return cv::countNonZero(diff);
        }

//#define UPDATE_TEST_DATA

#ifdef UPDATE_TEST_DATA
        TEST(Objdetect_QRCode_Decode, generate_test_data)
        {
            const std::string root = "qrcode/decode";
            const std::string dataset_config = findDataFile(root +"/"+ "dataset_config.json");
            FileStorage file_config(dataset_config, FileStorage::WRITE);

            file_config << "test_images" << "[";
            size_t images_count = sizeof(decode_qrcode_images_name) / sizeof(decode_qrcode_images_name[0]);
            for (size_t i = 0; i < images_count; i++)
            {
                file_config << "{:" << "image_name" << decode_qrcode_images_name[i];
                std::string image_path = findDataFile(root +"/"+ decode_qrcode_images_name[i]);

                /**read from test set*/
                Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
                std::vector<Point> src_corners(4);
                /**corners for src image(with 2 pixels border)*/
                src_corners[0] = Point(border_width,border_width);
                src_corners[1] = Point(src.rows - border_width , border_width);
                src_corners[2] = Point(src.rows - border_width , src.cols - border_width);
                src_corners[3] = Point(border_width , src.cols - border_width);
                /**pure qr-image without borders , which is enlarged to 600x600*/
                Mat src_no_border = src(Rect(src_corners[0],src_corners[2])).clone();
                resize(src_no_border,src_no_border,Size(600,600),0,0,INTER_AREA);

                std::string decoded_info;
                ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
                /**corners for QR image(without borders and fixed size of (600,600))*/
                std::vector<Point> decode_corners(4);
                decode_corners[0] = Point(0,0);
                decode_corners[1] = Point(src_no_border.rows - 1 , 0);
                decode_corners[2] = Point(src_no_border.rows - 1 , src_no_border.cols - 1);
                decode_corners[3] = Point(0 , src_no_border.cols - 1);

                EXPECT_TRUE(decodeQRCode(src_no_border, decode_corners, decoded_info, straight_barcode))<< "ERROR : " << image_path;

                file_config << "x" << "[:";
                for (size_t j = 0; j < src_corners.size(); j++) { file_config << src_corners[j].x; }
                file_config << "]";
                file_config << "y" << "[:";
                for (size_t j = 0; j < src_corners.size(); j++) { file_config << src_corners[j].y; }
                file_config << "]";
                /**use escape character for alternative interpretation in a character's sequence for “]” */
                size_t max_size = decoded_info.size();
                for(size_t t = 0 ; t <  max_size; t++ ){
                    if(decoded_info[t] == ']'||decoded_info[t] == '}'||decoded_info[t] == '['||decoded_info[t] == '{'){
                        decoded_info.insert(t,"\\");
                        t += 2;
                        max_size += 2;
                    }
                }
                file_config << "info" << decoded_info;
                file_config << "}";
            }
            file_config << "]";
            file_config.release();
        }

        TEST(Objdetect_QRCode_Encode, generate_test_data)
        {
            const std::string root = "qrcode/encode";
            const std::string dataset_config = findDataFile(root +"/"+ "dataset_config.json");
            FileStorage file_config(dataset_config, FileStorage::WRITE);

            file_config << "test_images" << "[";
            size_t images_count = sizeof(encode_qrcode_images_name) / sizeof(encode_qrcode_images_name[0]);
            for (size_t i = 0; i < images_count; i++)
            {
                file_config << "{:" << "image_name" << encode_qrcode_images_name[i];
                std::string image_path = findDataFile(root +"/"+ encode_qrcode_images_name[i]);

                /**read from test set*/
                Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
                src = src(Range(border_width,src.rows-border_width),Range(border_width,src.rows-border_width)).clone();
                resize(src,src,fixed_size,0,0,INTER_AREA);


                std::string decoded_info;
                ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

                /**add the corner points*/
                std::vector<Point> corners(4);
                corners[0] = Point(0,0);
                corners[1] = Point(src.rows-1,0);
                corners[2] = Point(src.rows-1,src.cols-1);
                corners[3] = Point(0,src.cols-1);

                EXPECT_TRUE(decodeQRCode(src, corners, decoded_info, straight_barcode))<< "ERROR : " << image_path;
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

        typedef testing::TestWithParam< std::string > Objdetect_QRCode_Decode;
        TEST_P(Objdetect_QRCode_Decode, regression)
        {
            const std::string name_current_image = GetParam();
            const std::string root = "qrcode/decode";

            std::string image_path = findDataFile(root +"/"+ name_current_image);
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
                        /**get corner points with 2-pixel borders*/
                        for (int i = 0; i < 4; i++)
                        {
                            int x = config["x"][i];
                            int y = config["y"][i];
                            corners.push_back(Point(x,y));
                        }
                        Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
                        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
                        /**pure qr-code without borders and with the size of 600x600*/
                        Mat src_no_border = src(Rect(corners[0],corners[2])).clone();
                        resize(src_no_border,src_no_border,fixed_size,0,0,INTER_AREA);
                        /**corners for QR image(without borders and fixed size of (600,600))*/
                        std::vector<Point> decode_corners(4);
                        decode_corners[0] = Point(0,0);
                        decode_corners[1] = Point(src_no_border.rows - 1 , 0);
                        decode_corners[2] = Point(src_no_border.rows - 1 , src_no_border.cols - 1);
                        decode_corners[3] = Point(0 , src_no_border.cols - 1);

                        EXPECT_TRUE(decodeQRCode(src_no_border, decode_corners, decoded_info, straight_barcode));
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
            const std::string root = "qrcode/encode";

            std::string image_path = findDataFile(root +"/"+ name_current_image);
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
                        /**encode a QR according to the string**/
                        QRCodeEncoder encoder;
                        Mat result ;
                        bool success = encoder.generate(original_info,result);
                        ASSERT_TRUE(success) << "Can't generate qr image :" << name_test_image;

                        /**read the original image */
                        Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
                        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

                        /**compare two qrcodes pixel by pixel*/
                        bool eq = countDiffPixels(result,src) == 0;//cv::countNonZero(diff) == 0;

                        /**the encoded info should be the same as the original info*/
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

        typedef testing::TestWithParam< std::string > Objdetect_QRCode_Encode_Decode;
        TEST_P(Objdetect_QRCode_Encode_Decode, regression){
            const std::string root = "qrcode/decode_encode";
            /**test the most popular version 1-7*/
            const int min_version = 1;
            const int test_max_version = 7;
            const int max_ecc = 3;
            const std::string dataset_config = findDataFile(root +"/"+ "symbol_sets.json");
            const std::string version_config =findDataFile(root +"/"+ "capacity.json");

            FileStorage file_config(dataset_config, FileStorage::READ);
            FileStorage capacity_config(version_config, FileStorage::READ);
            ASSERT_TRUE(file_config.isOpened()&&capacity_config.isOpened()) << "Can't read validation data: " << dataset_config;


            FileNode mode_list = file_config["symbols_sets"];
            FileNode capacity_list = capacity_config["version_ecc_capacity"];

            size_t mode_count = static_cast<size_t>(mode_list.size());
            ASSERT_GT(mode_count, 0u) << "Can't find validation data entries in 'test_images': " << dataset_config;

            for (size_t index = 0; index < mode_count; index++){
                /**loop each mode*/
                FileNode config = mode_list[(int)index];
                /**find corresponding symbol set*/
                std::string cur_mode = config["mode"];
                std::string symbol_set = config["symbols_set"];

                for(int j = min_version ; j <= test_max_version ; j ++ ){
                    /**Loop each version level (1:7)*/
                    FileNode capa_config = capacity_list[j-1];
                    for(int m = 0 ; m <= max_ecc ; m ++ ){
                        /**loop each ecc level*/
                        std::string cur_level = capa_config["verison_level"];
                        const int cur_capacity = capa_config["ecc_level"][m];

                        /**generate the input string **/
                        std::string input_info = symbol_set;
                        std::random_shuffle(input_info.begin(),input_info.end());
                        if((int)input_info.length() > cur_capacity){
                            input_info = input_info.substr(0,cur_capacity-1);
                        }

                        /**get the encoded QR codes */
                        QRCodeEncoder my_encoder;
                        vector<Mat> qrcodes;
                        bool generate_success = my_encoder.generate(input_info,qrcodes);
                        ASSERT_TRUE(generate_success) << "Can't generate this QR image :("<<"mode : "<<index<<
                                                      " version : "<<j<<" ecc_level : "<<m<<")";
                        std::string output_info = "";
                        for(size_t n = 0; n < qrcodes.size() ; n++ ){
                            Mat src = qrcodes[n];
                            src = src(Range(border_width,src.rows-border_width),Range(border_width,src.rows-border_width)).clone();
                            resize(src,src,Size(600,600),0,0,INTER_AREA);
                            /**set points and resize*/
                            std::vector<Point> corners(4);
                            corners[0] = Point(0,0);
                            corners[1] = Point(src.rows-1,0);
                            corners[2] = Point(src.rows-1,src.cols-1);
                            corners[3] = Point(0,src.cols-1);

                            std::string decoded_info ;
                            Mat straight_barcode;
                            bool success = decodeQRCode(src, corners, decoded_info, straight_barcode);
                            ASSERT_TRUE(success) << "The generated QRcode is not same as test data."<<"mode : "<<index<<
                                                 " version : "<<j<<" ecc_level : "<<m;
                            output_info += decoded_info;
                        }
                        ASSERT_TRUE(input_info == output_info) << "The generated QRcode is not same as test data."<<"mode : "<<index<<
                                                               " version : "<<j<<" ecc_level : "<<m;
                    }
                }
            }

        }

        INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Decode, testing::ValuesIn(decode_qrcode_images_name));
        INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Encode, testing::ValuesIn(encode_qrcode_images_name));
        INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Encode_Decode, testing::ValuesIn(encode_decode_qrcode_images_name));

#endif // UPDATE_QRCODE_TEST_DATA

    }} // namespace
