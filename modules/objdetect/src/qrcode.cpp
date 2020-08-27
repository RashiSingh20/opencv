// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"

#include <limits>
#include <cmath>
#include <iostream>
#include <queue>
#include <limits>
#include <iomanip>

namespace cv
{
/* Limits on the maximum size of QR-codes and their content. */
const int   max_payload_len = 8896;
const int max_format_length = 15;
const int  max_version = 40;
const int  max_alignment = 7;
const int  max_poly = 64;
const int  error_mode_occur = 99999;
/*for the reserved value when reading the data*/
const int invalid_region_value = 110;
const int  codeword_len = 8;


/**
 * Encoding mode.
 */
typedef enum {
    QR_MODE_NUL        = 0b0000,   ///< Terminator (NUL character). Internal use only
    QR_MODE_ECI        = 0b0111,        ///< ECI mode
    QR_MODE_NUM        = 0b0001,    ///< Numeric mode
    QR_MODE_ALPHA      = 0b0010,         ///< Alphabet-numeric mode
    QR_MODE_BYTE       = 0b0100,          ///< 8-bit data mode
    QR_MODE_KANJI      = 0b1000,      ///< Kanji (shift-jis) mode
    QR_MODE_STRUCTURE  = 0b0011,  ///< Internal use only
    QR_MODE_FNC1FIRST  = 0b0101, ///< FNC1, first position
    QR_MODE_FNC1SECOND = 0b1001, ///< FNC1, second position
} QRencodeMode;


enum ENCODING_SET {
    CP437 = 0,  // (Cp437 0)
    ISO_8859_1, // (ECI codes 1)
    CP437_,     // (Cp437 2
    ISO_8859_1_,// (ECI codes 3)
    ISO_8859_2, // (ECI code 4)
    ISO_8859_3, // (ECI code 5)
    ISO_8859_4, // (ECI code 6)
    ISO_8859_5, // (ECI code 7)
    ISO_8859_6, // (ECI code 8)
    ISO_8859_7, // (ECI code 9)
    ISO_8859_8, // (ECI code 10)
    ISO_8859_9, // (ECI code 11)
    ISO_8859_10, //(ECI code 12)
    ISO_8859_11, //(ECI code 13)
    ISO_8859_13 =15 , //(ECI code 15)
    ISO_8859_14, //(ECI code 16)
    ISO_8859_15, //(ECI code 17)
    ISO_8859_16, //(ECI code 18)
    Shift_JIS =20 ,   //(ECI code 20)
    CP1250,      // windows_1250,//(ECI code 21)
    CP1251,      // windows_1251,//(ECI code 22)
    CP1252,      // windows_1252,//(ECI code 23)
    CP1256,      // windows_1256,//(ECI code 24)
    UTF_16BE,    // UnicodeBig,UnicodeBigUnmarked,  (ECI code 25)
    UTF_8,       //(ECI code 26)
    US_ASCII,    //(ECI codes 27,170)
    Big5,        //(ECI code 28)
    GBK,         //GB18030, GB2312, EUC_CN, (ECI code 29)
    EUC_KR       //(ECI code 30)
};
using std::vector;

std::string decToBin(uint16_t my_format);
std::string decToBin(const int &format, const int &total);

int eccCodeToLevel(int code);
int eccLevelToCode(int level);

uint8_t gfPow(uint8_t x , int power);
uint8_t gfInverse(uint8_t x);
uint8_t gfMul(const uint8_t &x,const uint8_t& y);
uint8_t gfDiv(const uint8_t &x,const uint8_t& y);
uint8_t gfPolyEvaluate(const Mat& poly,uint8_t x);
Mat gfPolyScaling(const Mat & p,int scalar);
Mat gfPolyAdd(const Mat & p,const Mat & q);
Mat gfPolyMul(const Mat &p,const Mat &q);
Mat gfPolyDiv(const Mat& dividend,const Mat& divisor,const int& ecc_num);
Mat polyGenerator(const int & n );

Mat findErrorLocator(const vector<uint8_t>&synd,size_t & errors_len);
vector<int > findErrors(const Mat& sigma,const size_t &errors_len,const int & msg_len);
Mat errorCorrect(const Mat & msg_in ,const vector<uint8_t>&synd,const Mat & e_loc_poly,const vector<int> &error_index);

int hammingWeight(uint16_t x);
int hammingDetect(uint16_t fmt);

int calBlockSyndromes(const Mat & block, int synd_num,vector<uint8_t>& synd);
int getBits(const int& bits,const vector<uint8_t>& payload , int &pay_index);
void findAIofFNC1(const std::string & fnc1_AI, int & index,bool & is_find);

const char * getSrcMode(const int& eci_mode);
void loadString(vector<uint8_t> cur_str, int& len,const std::string &str ,bool is_bit_stream);


std::string decToBin(const int &format, const int &total){
    std::string f;
    int num = total;
    for(int i=format;num>0;i=i>>1,num--){
        if(i%2==1)
            f='1'+f;
        else
            f='0'+f;
    }
    return f;
}
const char * getSrcMode(const int& eci_mode){
    switch (eci_mode){
        case CP437:
        case CP437_:
            return "CP437";
        case ISO_8859_1:
        case ISO_8859_1_:
            return "ISO-8859-1";
        case ISO_8859_2:
            return "ISO-8859-2";
        case ISO_8859_3:
            return "ISO-8859-3";
        case ISO_8859_4:
            return "ISO-8859-4";
        case ISO_8859_5:
            return "ISO-8859-5";
        case ISO_8859_6:
            return "ISO-8859-6";
        case ISO_8859_7:
            return "ISO-8859-7";
        case ISO_8859_8:
            return "ISO-8859-8";
        case ISO_8859_9:
            return "ISO-8859-9";
        case ISO_8859_10:
            return "ISO-8859-10";
        case ISO_8859_11:
            return "ISO-8859-11";
        case ISO_8859_13:
            return "ISO-8859-13";
        case ISO_8859_14:
            return "ISO-8859-14";
        case ISO_8859_15:
            return "ISO-8859-15";
        case ISO_8859_16:
            return "ISO-8859-16";
        case Shift_JIS:
            return "SHIFT_JIS";
        case CP1250:
            return "CP1250";
        case CP1251:
            return "CP1251";
        case CP1252:
            return "CP1252";
        case CP1256:
            return "CP1256";
        case UTF_16BE:
            return "UTF−16BE";
        case UTF_8:
            return "UTF−8";
        case US_ASCII:
            return "ASCII";
        case Big5:
            return "BIG5";
        case GBK:
            return "GBK";
        case EUC_KR:
            return "EUC−KR";
        default:
            return "UTF−8";
    }
}
/**
 *params@  cur_str cur_str_len input str
 *func@ add a str into the cur_str
 */
void loadString(vector<uint8_t> cur_str,int& len,const std::string &str ,bool is_bit_stream = false){
    for(size_t i = 0; i < str.length();i++){
        if(is_bit_stream){
            cur_str[len++]=uint8_t (str[i]-'0');
        }
        else
            cur_str[len++]=uint8_t (str[i]);
    }
    return;
}

/*total codewords are divided into two groups
 *The ecc_codewords are the same in two groups*/
struct BlockParams {
    int ecc_codewords;           //Number of error correction blocks
    int num_blocks_in_G1;
    int data_codewords_in_G1;
    int num_blocks_in_G2;
    int data_codewords_in_G2;
};

struct VersionInfo {
    int	total_codewords;
    int	alignment_pattern[max_alignment];  //(location of alignment pattern)
    BlockParams ecc[4];
};
struct DataOfAI{
    int Data_len;
    bool fixed_len;
};
struct AIinGS1{
    std::string AI_name;
    int AI_len;
    DataOfAI data[2];
    std::string data_title;
};

const AIinGS1 GS1_AI_database[999] = {
        {"00",2,{{18,true},{0,0}},"\nSSCC"},
        {"01",2,{{14,true},{0,0}}, "\nGTIN"},// Global Trade Item Number
        {"02",2,{{14,true},{0,0}}, "\nCONTENT"},//GTIN of contained trade items
        {"10",2,{{20,false},{0,0}},"\nBATCH/LOT"},//Batch or lot number
        {"11",2,{{6 ,true},{0,0}}, "\nPROD DATE"},//Production date (YYMMDD)
        {"12",2,{{6 ,true},{0,0}},"\nDUE DATE"},//Due date (YYMMDD) N2+N6
        {"13",2,{{6 ,true},{0,0}},"\nPACK DATE"},// Packaging date (YYMMDD)
        {"15",2,{{6 ,true},{0,0}}, "\nBEST BEFORE"},//Best before date (YYMMDD)
        {"16",2,{{6 ,true},{0,0}},"\nSELL BY"},//Sell by date (YYMMDD)
        {"17",2,{{6 ,true},{0,0}}, "\nEXPIRY"},//Expiration date (YYMMDD)
        {"20",2,{{2 ,true},{0,0}}, "\nVARIANT"},//Variant number
        {"21",2,{{20,false},{0,0}}, "\nSERIAL"},// Serial number
        {"240",3,{{30,false},{0,0}}, "\nADDITIONAL ID"},// Additional item identification
        {"241",3,{{30,false},{0,0}}, "\nCUST. PART NO."},// Customer part number
        {"242",3,{{6,false},{0,0}}, "\nMTO VARIANT"},// Made-to-Order variation number
        {"243",3,{{20,false},{0,0}}, "\nPCN"},// Packaging component number
        {"250",3,{{30,false},{0,0}},"\nSECONDARY SERIAL"},// Secondary serial number
        {"251",3,{{30,false},{0,0}},"\nREF. TO SOURCE"},//Reference to source entity
        {"253",3,{{13,true},{17,false}},"\nGDTI"},// Global Document Type Identifier (GDTI)
        {"254",3,{{20,false},{0,0}},"\nGLN EXTENSION COMPONENT"},// GLN extension component
        {"255",3,{{13,true},{12,false}},"\nGCN"},// Global Coupon Number (GCN)
        {"30" ,2,{{8, false},{0,0}},"\nVAR. COUNT"},// Count of items (variable measure trade item)
        {"8200",4,{{70,false},{0,0}},"\nPRODUCT URL"}// Extended Packaging URL
};


const VersionInfo version_info_database[max_version + 1] = {
        { /* Version 0 */
                0,
                {0,0,0,0,0,0,0},
                {
                        {0	,0	,0,0,0},
                        {0	,0	,0,0,0},
                        {0	,0	,0,0,0},
                        {0	,0	,0 ,0,0}
                }
        },
        { /* Version 1 */
                26,
                {0,0,0,0,0,0,0},
                {
                        {7	,1	,19,0,0},
                        {10	,1	,16,0,0},
                        {13	,1	,13,0,0},
                        {17	,1	,9 ,0,0}
                }
        },
        { /* Version 2 */
                44,
                {6, 18, 0,0,0,0,0},
                {
                        { 10,	1,	34,0,0},
                        {  16,	1,	28,0,0},
                        {  22,	1,	22,0,0},
                        {  28,	1,	16,0,0}
                }
        },
        { /* Version 3 */
                70,
                {6, 22, 0,0,0,0,0},
                {
                        {  15,	1,	55,0,0},
                        {  26,	1,	44,0,0},
                        {  18,	2,	17,0,0},
                        {  22,	2,	13,0,0}
                }
        },
        { /* Version 4 */
                100,
                {6, 26, 0,0,0,0,0},
                {
                        {  20,	1,	80,0,0},
                        {  18,	2,	32,0,0},
                        {  26,	2,	24,0,0},
                        {  16,	4,	9 ,0,0}
                }
        },
        { /* Version 5 */
                134,
                {6, 30, 0,0,0,0,0},
                {
                        {  26,	1,	108,0,  0},
                        {  24,	2,	43, 0,  0},
                        {  18,	2,	15,	2,	16},
                        {  22,	2,	11,	2,	12}
                }
        },
        { /* Version 6 */
                172,
                {6, 34, 0,0,0,0,0},
                {
                        {  18,	2,	68,0,0},
                        {  16,	4,	27,0,0},
                        {  24,	4,	19,0,0},
                        {  28,	4,	15,0,0}
                }
        },
        { /* Version 7 */
                196,
                {6, 22, 38, 0,0,0,0},
                {
                        {  20,	2,	78,0,0},
                        {  18,	4,	31,0,0},
                        {  18,	2,	14,	4,	15},
                        {  26,	4,	13,	1,	14}
                }
        },
        { /* Version 8 */
                242,
                {6, 24, 42, 0,0,0,0},
                {
                        {  24,	2,	97,0,0},
                        {  22,	2,	38,	2,	39},
                        {  22,	4,	18,	2,	19},
                        {  26,	4,	14,	2,	15}
                }
        },
        { /* Version 9 */
                292,
                {6, 26, 46, 0,0,0,0},
                {
                        {  30,	2,	116,0,0},
                        {  22,	3,	36,	2,	37},
                        {  20,	4,	16,	4,	17},
                        {  24,	4,	12,	4,	13}
                }
        },
        { /* Version 10 */
                346,
                {6, 28, 50, 0,0,0,0},
                {
                        {  18,	2,	68,	2,	69},
                        {  26,	4,	43,	1,	44},
                        {  24,	6,	19,	2,	20},
                        {  28,	6,	15,	2,	16}
                }
        },
        { /* Version 11 */
                404,
                {6, 30, 54, 0,0,0,0},
                {
                        {  20,	4,	81, 0,0},
                        {  30,	1,	50,	4,	51},
                        {  28,	4,	22,	4,	23},
                        {  24,	3,	12,	8,	13}
                }
        },
        { /* Version 12 */
                466,
                {6, 32, 58, 0,0,0,0},
                {
                        {  24,	2,	92,	2,	93},
                        {  22,	6,	36,	2,	37},
                        {  26,	4,	20,	6,	21},
                        {  28,	7,	14,	4,	15}
                }
        },
        { /* Version 13 */
                532,
                {6, 34, 62, 0,0,0,0},
                {
                        {  26,	4,	107,0,0},
                        {  22,	8,	37,	1,	38},
                        {  24,	8,	20,	4,	21},
                        {  22,	12,	11,	4,	12}
                }
        },
        { /* Version 14 */
                581,
                {6, 26, 46, 66, 0,0,0},
                {
                        {  30,	3,	115,1,	116},
                        {  24,	4,	40,	5,	41},
                        {  20,	11,	16,	5,	17},
                        {  24,	11,	12,	5,	13}
                }
        },
        { /* Version 15 */
                655,
                {6, 26, 48, 70, 0,0,0},
                {
                        {  22,	5,	87,	1,	88},
                        {  24,	5,	41,	5,	42},
                        {  30,	5,	24,	7,	25},
                        {  24,	11,	12,	7,	13}
                }
        },
        { /* Version 16 */
                733,
                {6, 26, 50, 74, 0,0,0},
                {
                        {  24,	5,	98,	1,	99},
                        {  28,	7,	45,	3,	46},
                        {  24,	15,	19,	2,	20},
                        {  30,	3,	15,	13,	16}
                }
        },
        { /* Version 17 */
                815,
                {6, 30, 54, 78, 0,0,0},
                {
                        { 28,	1,	107,5,	108},
                        { 28,	10,	46,	1,	47},
                        { 28,	1,	22,	15,	23},
                        { 28,	2,	14,	17,	15}
                }
        },
        { /* Version 18 */
                901,
                {6, 30, 56, 82, 0,0,0},
                {
                        {  30,	5,	120,1,	121},
                        {  26,	9,	43,	4,	44},
                        {  28,	17,	22,	1,	23},
                        {  28,	2,	14,	19,	15}
                }
        },
        { /* Version 19 */
                991,
                {6, 30, 58, 86, 0,0,0},
                {
                        {  28,	3,	113,4,	114},
                        {  26,	3,	44,	11,	45},
                        {  26,	17,	21,	4,	22},
                        {  26,	9,	13,	16,	14}
                }
        },
        { /* Version 20 */
                1085,
                {6, 34, 62, 90, 0,0,0},
                {
                        {  28,	3,	107,5,	108},
                        {  26,	3,	41,	13,	42},
                        {  30,	15,	24,	5,	25},
                        {  28,	15,	15,	10,	16}
                }
        },
        { /* Version 21 */
                1156,
                {6, 28, 50, 72, 92, 0,0},
                {
                        {  28,	4,	116,4,  117},
                        {  26,	17,	42, 0,  0},
                        {  28,	17,	22,	6,	23},
                        {  30,	19,	16,	6,	17}
                }
        },
        { /* Version 22 */
                1258,
                {6, 26, 50, 74, 98, 0,0},
                {
                        {  28,	2,	111, 7,	 112},
                        {  28,	17,	46,  0,  0},
                        {  30,	7,	24,	 16, 25},
                        {  24,	34,	13,  0,  0}
                }
        },
        { /* Version 23 */
                1364,
                {6, 30, 54, 78, 102, 0,0},
                {
                        {  30,	4,	121,5,	122},
                        {  28,	4,	47,	14,	48},
                        {  30,	11,	24,	14,	25},
                        {  30,	16,	15,	14,	16}
                }
        },
        { /* Version 24 */
                1474,
                {6, 28, 54, 80, 106, 0,0},
                {
                        {  30,	6,	117,4,	118},
                        {  28,	6,	45,	14,	46},
                        {  30,	11,	24,	16,	25},
                        {  30,	30,	16,	2,	17}
                }
        },
        { /* Version 25 */
                1588,
                {6, 32, 58, 84, 110, 0,0},
                {
                        {  26,	8,	106,4,	107},
                        {  28,	8,	47,	13,	48},
                        {  30,	7,	24,	22,	25},
                        {  30,	22,	15,	13,	16}
                }
        },
        { /* Version 26 */
                1706,
                {6, 30, 58, 86, 114, 0,0},
                {
                        {  28,	10,	114,2,	115},
                        {  28,	19,	46,	4,	47},
                        {  28,	28,	22,	6,	23},
                        {  30,	33,	16,	4,	17}
                }
        },
        { /* Version 27 */
                1828,
                {6, 34, 62, 90, 118, 0,0},
                {
                        {  30,	8,	122,4,	123},
                        {  28,	22,	45,	3,	46},
                        {  30,	8,	23,	26,	24},
                        {  30,	12,	15,	28,	16}
                }
        },
        { /* Version 28 */
                1921,
                {6, 26, 50, 74, 98, 122, 0},
                {
                        {  30,	3,	117,10,	118},
                        {  28,	3,	45,	23,	46},
                        {  30,	4,	24,	31,	25},
                        {  30,	11,	15,	31,	16}
                }
        },
        { /* Version 29 */
                2051,
                {6, 30, 54, 78, 102, 126, 0},
                {
                        {  30,	7,	116,7,	117},
                        {  28,	21,	45,	7,	46},
                        {  30,	1,	23,	37,	24},
                        {  30,	19,	15,	26,	16}
                }
        },
        { /* Version 30 */
                2185,
                {6, 26, 52, 78, 104, 130, 0},
                {
                        {  30,	5,	115,10,	116},
                        {  28,	19,	47,	10,	48},
                        {  30,	15,	24,	25,	25},
                        {  30,	23,	15,	25,	16}
                }
        },
        { /* Version 31 */
                2323,
                {6, 30, 56, 82, 108, 134, 0},
                {
                        {  30,	13,	115,3,	116},
                        {  28,	2,	46,	29,	47},
                        {  30,	42,	24,	1,	25},
                        {  30,	23,	15,	28,	16}
                }
        },
        { /* Version 32 */
                2465,
                {6, 34, 60, 86, 112, 138, 0},
                {
                        {  30,	17,	115, 0,     0},
                        {  28,	10,	46,	 23,	47},
                        {  30,	10,	24,	 35,	25},
                        {  30,	19,	15,	 35,	16}
                }
        },
        { /* Version 33 */
                2611,
                {6, 30, 58, 86, 114, 142, 0},
                {
                        {  30,	17,	115,1,	116},
                        {  28,	14,	46,	21,	47},
                        {  30,	29,	24,	19,	25},
                        {  30,	11,	15,	46,	16}
                }
        },
        { /* Version 34 */
                2761,
                {6, 34, 62, 90, 118, 146, 0},
                {
                        {  30,	13,	115,6,	116},
                        {  28,	14,	46,	23,	47},
                        {  30,	44,	24,	7,	25},
                        {  30,	59,	16,	1,	17}
                }
        },
        { /* Version 35 */
                2876,
                {6, 30, 54, 78, 102, 126, 150},
                {
                        {  30,	12,	121,7,	122},
                        {  28,	12,	47,	26,	48},
                        {  30,	39,	24,	14,	25},
                        {  30,	22,	15,	41,	16}
                }
        },
        { /* Version 36 */
                3034,
                {6, 24, 50, 76, 102, 128, 154},
                {
                        {  30,	6,	121,14,	122},
                        {  28,	6,	47,	34,	48},
                        {  30,	46,	24,	10,	25},
                        {  30,	2,	15,	64,	16}
                }
        },
        { /* Version 37 */
                3196,
                {6, 28, 54, 80, 106, 132, 158},
                {
                        {  30,	17,	122,4,	123},
                        {  28,	29,	46,	14,	47},
                        {  30,	49,	24,	10,	25},
                        {  30,	24,	15,	46,	16}
                }
        },
        { /* Version 38 */
                3362,
                {6, 32, 58, 84, 110, 136, 162},
                {
                        {  30,	4,	122,18,	123},
                        {  28,	13,	46,	32,	47},
                        {  30,	48,	24,	14,	25},
                        {  30,	42,	15,	32,	16}
                }
        },
        { /* Version 39 */
                3532,
                {6, 26, 54, 82, 110, 138, 166},
                {
                        {  30,	20,	117,4,	118},
                        {  28,	40,	47,	7,	48},
                        {  30,	43,	24,	22,	25},
                        {  30,	10,	15,	67,	16}
                }
        },
        { /* Version 40 */
                3706,
                {6, 30, 58, 86, 114, 142, 170},
                {
                        {  30,	19,	118,6,	119},
                        {  28,	18,	47,	31,	48},
                        {  30,	34,	24,	34,	25},
                        {  30,	20,	15,	61,	16}
                }
        }
};
static const uint16_t after_mask_format [32]={
        0x5412,0x5125,0x5e7c,0x5b4b,0x45f9,  0x40ce,0x4f97,0x4aa0,0x77c4,0x72f3,
        0x7daa,0x789d,0x662f,0x6318,0x6c41,  0x6976,0x1689,0x13be,0x1ce7,0x19d0,
        0x0762,0x0255,0x0d0c,0x083b,0x355f,  0x3068,0x3f31,0x3a06,0x24b4,0x2183,
        0x2eda,0x2bed
};

static const uint8_t gf_exp[256] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26,
        0x4c, 0x98, 0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9,
        0x8f, 0x03, 0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0,
        0x9d, 0x27, 0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35,
        0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23,
        0x46, 0x8c, 0x05, 0x0a, 0x14, 0x28, 0x50, 0xa0,
        0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1,
        0x5f, 0xbe, 0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc,
        0x65, 0xca, 0x89, 0x0f, 0x1e, 0x3c, 0x78, 0xf0,
        0xfd, 0xe7, 0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f,
        0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2,
        0xd9, 0xaf, 0x43, 0x86, 0x11, 0x22, 0x44, 0x88,
        0x0d, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce,
        0x81, 0x1f, 0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93,
        0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc,
        0x85, 0x17, 0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9,
        0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54,
        0xa8, 0x4d, 0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa,
        0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73,
        0xe6, 0xd1, 0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e,
        0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff,
        0xe3, 0xdb, 0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4,
        0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41,
        0x82, 0x19, 0x32, 0x64, 0xc8, 0x8d, 0x07, 0x0e,
        0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6,
        0x51, 0xa2, 0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef,
        0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x09,
        0x12, 0x24, 0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5,
        0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0x0b, 0x16,
        0x2c, 0x58, 0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83,
        0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x01
};
static const uint8_t gf_log[256] = {
        0x00, 0xff, 0x01, 0x19, 0x02, 0x32, 0x1a, 0xc6,
        0x03, 0xdf, 0x33, 0xee, 0x1b, 0x68, 0xc7, 0x4b,
        0x04, 0x64, 0xe0, 0x0e, 0x34, 0x8d, 0xef, 0x81,
        0x1c, 0xc1, 0x69, 0xf8, 0xc8, 0x08, 0x4c, 0x71,
        0x05, 0x8a, 0x65, 0x2f, 0xe1, 0x24, 0x0f, 0x21,
        0x35, 0x93, 0x8e, 0xda, 0xf0, 0x12, 0x82, 0x45,
        0x1d, 0xb5, 0xc2, 0x7d, 0x6a, 0x27, 0xf9, 0xb9,
        0xc9, 0x9a, 0x09, 0x78, 0x4d, 0xe4, 0x72, 0xa6,
        0x06, 0xbf, 0x8b, 0x62, 0x66, 0xdd, 0x30, 0xfd,
        0xe2, 0x98, 0x25, 0xb3, 0x10, 0x91, 0x22, 0x88,
        0x36, 0xd0, 0x94, 0xce, 0x8f, 0x96, 0xdb, 0xbd,
        0xf1, 0xd2, 0x13, 0x5c, 0x83, 0x38, 0x46, 0x40,
        0x1e, 0x42, 0xb6, 0xa3, 0xc3, 0x48, 0x7e, 0x6e,
        0x6b, 0x3a, 0x28, 0x54, 0xfa, 0x85, 0xba, 0x3d,
        0xca, 0x5e, 0x9b, 0x9f, 0x0a, 0x15, 0x79, 0x2b,
        0x4e, 0xd4, 0xe5, 0xac, 0x73, 0xf3, 0xa7, 0x57,
        0x07, 0x70, 0xc0, 0xf7, 0x8c, 0x80, 0x63, 0x0d,
        0x67, 0x4a, 0xde, 0xed, 0x31, 0xc5, 0xfe, 0x18,
        0xe3, 0xa5, 0x99, 0x77, 0x26, 0xb8, 0xb4, 0x7c,
        0x11, 0x44, 0x92, 0xd9, 0x23, 0x20, 0x89, 0x2e,
        0x37, 0x3f, 0xd1, 0x5b, 0x95, 0xbc, 0xcf, 0xcd,
        0x90, 0x87, 0x97, 0xb2, 0xdc, 0xfc, 0xbe, 0x61,
        0xf2, 0x56, 0xd3, 0xab, 0x14, 0x2a, 0x5d, 0x9e,
        0x84, 0x3c, 0x39, 0x53, 0x47, 0x6d, 0x41, 0xa2,
        0x1f, 0x2d, 0x43, 0xd8, 0xb7, 0x7b, 0xa4, 0x76,
        0xc4, 0x17, 0x49, 0xec, 0x7f, 0x0c, 0x6f, 0xf6,
        0x6c, 0xa1, 0x3b, 0x52, 0x29, 0x9d, 0x55, 0xaa,
        0xfb, 0x60, 0x86, 0xb1, 0xbb, 0xcc, 0x3e, 0x5a,
        0xcb, 0x59, 0x5f, 0xb0, 0x9c, 0xa9, 0xa0, 0x51,
        0x0b, 0xf5, 0x16, 0xeb, 0x7a, 0x75, 0x2c, 0xd7,
        0x4f, 0xae, 0xd5, 0xe9, 0xe6, 0xe7, 0xad, 0xe8,
        0x74, 0xd6, 0xf4, 0xea, 0xa8, 0x50, 0x58, 0xaf
};

/*
 * params @ ecc_level
 * func   @ make the ecc_level more direct
 * return @ the actual value of ecc_level
 * */
int eccCodeToLevel(int code){
    switch (code){
        case 0b01://L	01
            return 0;
        case 0b00 ://M	00
            return 1;
        case 0b11 ://Q	11
            return 2;
        case 0b10 ://H	10
            return 3;
    }
    return 0;
}
/*
 * params @ eccLevelToCode
 * func   @ make the ecc_level more direct
 * return @ the actual value of ecc_code
 * */
int eccLevelToCode(int level){
    switch (level){
        case 0://L	01
            return 0b01;
        case 1 ://M	00
            return 0b00;
        case 2 ://Q	11
            return 0b11;
        case 3 ://H	10
            return 0b10;
    }
    return -1;
}
/*
 * params @ a number
 * func   @ convert a dec to bin
 * return @ a bin string for print
 * */

std::string decToBin(uint16_t my_format){
    std::string f;
    for(int i=my_format;i>0;i=i>>1){
        if(i%2==1)
            f='1'+f;
        else
            f='0'+f;
    }
    return f;
}

/*hammingWeight:
 *      get the distance by counting the number of 1
 * params @ uint16_t x(input the XOR of two binary string)
 * return @ the distance
 */
int hammingWeight(uint16_t x){
    int weight=0;
    while(x>0){
        weight += x & 1;
        x >>= 1;
    }
    return weight;
}

/*hammingDetect:
 *      find the best matched string
 * params @ uint16_t fmt(input format)
 * return @ the index of matched component in the look-up table
*/

int hammingDetect(uint16_t fmt){
    int best_fmt = -1;
    int best_dist = 15;
    int test_dist;
    for(int index=0;index<32;index++){
        /**fetch out from the table*/
        uint16_t test_code=after_mask_format[index];
        test_dist=hammingWeight(fmt ^ test_code);
        /**find the smallest distance*/
        if (test_dist < best_dist){
            best_dist = test_dist;
            best_fmt = index;
        }
        /**there can't be two distance with the same value*/
        else if(test_dist == best_dist) {
            best_fmt = -1;
        }
    }
    return best_fmt;
}

static bool checkQRInputImage(InputArray img, Mat& gray)
{
    CV_Assert(!img.empty());
    CV_CheckDepthEQ(img.depth(), CV_8U, "");

    if (img.cols() <= 20 || img.rows() <= 20)
    {
        return false;  // image data is not enough for providing reliable results
    }
    int incn = img.channels();
    CV_Check(incn, incn == 1 || incn == 3 || incn == 4, "");
    if (incn == 3 || incn == 4)
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = img.getMat();
    }
    return true;
}

static void updatePointsResult(OutputArray points_, const vector<Point2f>& points)
{
    if (points_.needed())
    {
        int N = int(points.size() / 4);
        if (N > 0)
        {
            Mat m_p(N, 4, CV_32FC2, (void*)&points[0]);
            int points_type = points_.fixedType() ? points_.type() : CV_32FC2;
            m_p.reshape(2, points_.rows()).convertTo(points_, points_type);  // Mat layout: N x 4 x 2cn
        }
        else
        {
            points_.release();
        }
    }
}



class QRDetect
{
public:
    void init(const Mat& src, double eps_vertical_ = 0.2, double eps_horizontal_ = 0.1);
    bool localization();
    bool computeTransformationPoints();
    Mat getBinBarcode() { return bin_barcode; }
    Mat getStraightBarcode() { return straight_barcode; }
    vector<Point2f> getTransformationPoints() { return transformation_points; }
    static Point2f intersectionLines(Point2f a1, Point2f a2, Point2f b1, Point2f b2);
protected:
    vector<Vec3d> searchHorizontalLines();
    vector<Point2f> separateVerticalLines(const vector<Vec3d> &list_lines);
    vector<Point2f> extractVerticalLines(const vector<Vec3d> &list_lines, double eps);
    void fixationPoints(vector<Point2f> &local_point);
    vector<Point2f> getQuadrilateral(vector<Point2f> angle_list);
    bool testBypassRoute(vector<Point2f> hull, int start, int finish);
    inline double getCosVectors(Point2f a, Point2f b, Point2f c);

    Mat barcode, bin_barcode, resized_barcode, resized_bin_barcode, straight_barcode;
    vector<Point2f> localization_points, transformation_points;
    double eps_vertical, eps_horizontal, coeff_expansion;
    enum resize_direction { ZOOMING, SHRINKING, UNCHANGED } purpose;
};


void QRDetect::init(const Mat& src, double eps_vertical_, double eps_horizontal_)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!src.empty());
    barcode = src.clone();
    const double min_side = std::min(src.size().width, src.size().height);
    if (min_side < 512.0)
    {
        purpose = ZOOMING;
        coeff_expansion = 512.0 / min_side;
        const int width  = cvRound(src.size().width  * coeff_expansion);
        const int height = cvRound(src.size().height  * coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_LINEAR);
    }
    else if (min_side > 512.0)
    {
        purpose = SHRINKING;
        coeff_expansion = min_side / 512.0;
        const int width  = cvRound(src.size().width  / coeff_expansion);
        const int height = cvRound(src.size().height  / coeff_expansion);
        Size new_size(width, height);
        resize(src, resized_barcode, new_size, 0, 0, INTER_AREA);
    }
    else
    {
        purpose = UNCHANGED;
        coeff_expansion = 1.0;
    }

    eps_vertical   = eps_vertical_;
    eps_horizontal = eps_horizontal_;

    if (!barcode.empty())
        adaptiveThreshold(barcode, bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    else
        bin_barcode.release();

    if (!resized_barcode.empty())
        adaptiveThreshold(resized_barcode, resized_bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    else
        resized_bin_barcode.release();
}

vector<Vec3d> QRDetect::searchHorizontalLines()
{
    CV_TRACE_FUNCTION();
    vector<Vec3d> result;
    const int height_bin_barcode = bin_barcode.rows;
    const int width_bin_barcode  = bin_barcode.cols;
    const size_t test_lines_size = 5;
    double test_lines[test_lines_size];
    vector<size_t> pixels_position;

    for (int y = 0; y < height_bin_barcode; y++)
    {
        pixels_position.clear();
        const uint8_t *bin_barcode_row = bin_barcode.ptr<uint8_t>(y);

        int pos = 0;
        for (; pos < width_bin_barcode; pos++) { if (bin_barcode_row[pos] == 0) break; }
        if (pos == width_bin_barcode) { continue; }

        pixels_position.push_back(pos);
        pixels_position.push_back(pos);
        pixels_position.push_back(pos);

        uint8_t future_pixel = 255;
        for (int x = pos; x < width_bin_barcode; x++)
        {
            if (bin_barcode_row[x] == future_pixel)
            {
                future_pixel = static_cast<uint8_t>(~future_pixel);
                pixels_position.push_back(x);
            }
        }
        pixels_position.push_back(width_bin_barcode - 1);
        for (size_t i = 2; i < pixels_position.size() - 4; i+=2)
        {
            test_lines[0] = static_cast<double>(pixels_position[i - 1] - pixels_position[i - 2]);
            test_lines[1] = static_cast<double>(pixels_position[i    ] - pixels_position[i - 1]);
            test_lines[2] = static_cast<double>(pixels_position[i + 1] - pixels_position[i    ]);
            test_lines[3] = static_cast<double>(pixels_position[i + 2] - pixels_position[i + 1]);
            test_lines[4] = static_cast<double>(pixels_position[i + 3] - pixels_position[i + 2]);

            double length = 0.0, weight = 0.0;  // TODO avoid 'double' calculations

            for (size_t j = 0; j < test_lines_size; j++) { length += test_lines[j]; }

            if (length == 0) { continue; }
            for (size_t j = 0; j < test_lines_size; j++)
            {
                if (j != 2) { weight += fabs((test_lines[j] / length) - 1.0/7.0); }
                else        { weight += fabs((test_lines[j] / length) - 3.0/7.0); }
            }

            if (weight < eps_vertical)
            {
                Vec3d line;
                line[0] = static_cast<double>(pixels_position[i - 2]);
                line[1] = y;
                line[2] = length;
                result.push_back(line);
            }
        }
    }
    return result;
}

vector<Point2f> QRDetect::separateVerticalLines(const vector<Vec3d> &list_lines)
{
    CV_TRACE_FUNCTION();

    for (int coeff_epsilon = 1; coeff_epsilon < 10; coeff_epsilon++)
    {
        vector<Point2f> point2f_result = extractVerticalLines(list_lines, eps_horizontal * coeff_epsilon);
        if (!point2f_result.empty())
        {
            vector<Point2f> centers;
            Mat labels;
            double compactness = kmeans(
                    point2f_result, 3, labels,
                    TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
                    3, KMEANS_PP_CENTERS, centers);
            if (compactness == 0)
                continue;
            if (compactness > 0)
            {
                return point2f_result;
            }
        }
    }
    return vector<Point2f>();  // nothing
}

vector<Point2f> QRDetect::extractVerticalLines(const vector<Vec3d> &list_lines, double eps)
{
    CV_TRACE_FUNCTION();
    vector<Vec3d> result;
    vector<double> test_lines; test_lines.reserve(6);

    for (size_t pnt = 0; pnt < list_lines.size(); pnt++)
    {
        const int x = cvRound(list_lines[pnt][0] + list_lines[pnt][2] * 0.5);
        const int y = cvRound(list_lines[pnt][1]);

        // --------------- Search vertical up-lines --------------- //

        test_lines.clear();
        uint8_t future_pixel_up = 255;

        int temp_length_up = 0;
        for (int j = y; j < bin_barcode.rows - 1; j++)
        {
            uint8_t next_pixel = bin_barcode.ptr<uint8_t>(j + 1)[x];
            temp_length_up++;
            if (next_pixel == future_pixel_up)
            {
                future_pixel_up = static_cast<uint8_t>(~future_pixel_up);
                test_lines.push_back(temp_length_up);
                temp_length_up = 0;
                if (test_lines.size() == 3)
                    break;
            }
        }

        // --------------- Search vertical down-lines --------------- //

        int temp_length_down = 0;
        uint8_t future_pixel_down = 255;
        for (int j = y; j >= 1; j--)
        {
            uint8_t next_pixel = bin_barcode.ptr<uint8_t>(j - 1)[x];
            temp_length_down++;
            if (next_pixel == future_pixel_down)
            {
                future_pixel_down = static_cast<uint8_t>(~future_pixel_down);
                test_lines.push_back(temp_length_down);
                temp_length_down = 0;
                if (test_lines.size() == 6)
                    break;
            }
        }

        // --------------- Compute vertical lines --------------- //

        if (test_lines.size() == 6)
        {
            double length = 0.0, weight = 0.0;  // TODO avoid 'double' calculations

            for (size_t i = 0; i < test_lines.size(); i++)
                length += test_lines[i];

            CV_Assert(length > 0);
            for (size_t i = 0; i < test_lines.size(); i++)
            {
                if (i % 3 != 0)
                {
                    weight += fabs((test_lines[i] / length) - 1.0/ 7.0);
                }
                else
                {
                    weight += fabs((test_lines[i] / length) - 3.0/14.0);
                }
            }

            if (weight < eps)
            {
                result.push_back(list_lines[pnt]);
            }
        }
    }

    vector<Point2f> point2f_result;
    if (result.size() > 2)
    {
        for (size_t i = 0; i < result.size(); i++)
        {
            point2f_result.push_back(
                  Point2f(static_cast<float>(result[i][0] + result[i][2] * 0.5),
                          static_cast<float>(result[i][1])));
        }
    }
    return point2f_result;
}

void QRDetect::fixationPoints(vector<Point2f> &local_point)
{
    CV_TRACE_FUNCTION();
    double cos_angles[3], norm_triangl[3];

    norm_triangl[0] = norm(local_point[1] - local_point[2]);
    norm_triangl[1] = norm(local_point[0] - local_point[2]);
    norm_triangl[2] = norm(local_point[1] - local_point[0]);

    cos_angles[0] = (norm_triangl[1] * norm_triangl[1] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[0] * norm_triangl[0]) / (2 * norm_triangl[1] * norm_triangl[2]);
    cos_angles[1] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[1] * norm_triangl[1]) / (2 * norm_triangl[0] * norm_triangl[2]);
    cos_angles[2] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[1] * norm_triangl[1]
                  -  norm_triangl[2] * norm_triangl[2]) / (2 * norm_triangl[0] * norm_triangl[1]);

    const double angle_barrier = 0.85;
    if (fabs(cos_angles[0]) > angle_barrier || fabs(cos_angles[1]) > angle_barrier || fabs(cos_angles[2]) > angle_barrier)
    {
        local_point.clear();
        return;
    }

    size_t i_min_cos =
       (cos_angles[0] < cos_angles[1] && cos_angles[0] < cos_angles[2]) ? 0 :
       (cos_angles[1] < cos_angles[0] && cos_angles[1] < cos_angles[2]) ? 1 : 2;

    size_t index_max = 0;
    double max_area = std::numeric_limits<double>::min();
    for (size_t i = 0; i < local_point.size(); i++)
    {
        const size_t current_index = i % 3;
        const size_t left_index  = (i + 1) % 3;
        const size_t right_index = (i + 2) % 3;

        const Point2f current_point(local_point[current_index]),
            left_point(local_point[left_index]), right_point(local_point[right_index]),
            central_point(intersectionLines(current_point,
                              Point2f(static_cast<float>((local_point[left_index].x + local_point[right_index].x) * 0.5),
                                      static_cast<float>((local_point[left_index].y + local_point[right_index].y) * 0.5)),
                              Point2f(0, static_cast<float>(bin_barcode.rows - 1)),
                              Point2f(static_cast<float>(bin_barcode.cols - 1),
                                      static_cast<float>(bin_barcode.rows - 1))));


        vector<Point2f> list_area_pnt;
        list_area_pnt.push_back(current_point);

        vector<LineIterator> list_line_iter;
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, left_point));
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, central_point));
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, right_point));

        for (size_t k = 0; k < list_line_iter.size(); k++)
        {
            LineIterator& li = list_line_iter[k];
            uint8_t future_pixel = 255, count_index = 0;
            for(int j = 0; j < li.count; j++, ++li)
            {
                const Point p = li.pos();
                if (p.x >= bin_barcode.cols ||
                    p.y >= bin_barcode.rows)
                {
                    break;
                }

                const uint8_t value = bin_barcode.at<uint8_t>(p);
                if (value == future_pixel)
                {
                    future_pixel = static_cast<uint8_t>(~future_pixel);
                    count_index++;
                    if (count_index == 3)
                    {
                        list_area_pnt.push_back(p);
                        break;
                    }
                }
            }
        }

        const double temp_check_area = contourArea(list_area_pnt);
        if (temp_check_area > max_area)
        {
            index_max = current_index;
            max_area = temp_check_area;
        }

    }
    if (index_max == i_min_cos) { std::swap(local_point[0], local_point[index_max]); }
    else { local_point.clear(); return; }

    const Point2f rpt = local_point[0], bpt = local_point[1], gpt = local_point[2];
    Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
    if( determinant(m) > 0 )
    {
        std::swap(local_point[1], local_point[2]);
    }
}

bool QRDetect::localization()
{
    CV_TRACE_FUNCTION();
    Point2f begin, end;
    vector<Vec3d> list_lines_x = searchHorizontalLines();
    if( list_lines_x.empty() ) { return false; }
    vector<Point2f> list_lines_y = separateVerticalLines(list_lines_x);
    if( list_lines_y.empty() ) { return false; }

    vector<Point2f> centers;
    Mat labels;
    kmeans(list_lines_y, 3, labels,
           TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
           3, KMEANS_PP_CENTERS, localization_points);

    fixationPoints(localization_points);

    bool suare_flag = false, local_points_flag = false;
    double triangle_sides[3];
    double triangle_perim, square_area, img_square_area;
    if (localization_points.size() == 3)
    {
        triangle_sides[0] = norm(localization_points[0] - localization_points[1]);
        triangle_sides[1] = norm(localization_points[1] - localization_points[2]);
        triangle_sides[2] = norm(localization_points[2] - localization_points[0]);

        triangle_perim = (triangle_sides[0] + triangle_sides[1] + triangle_sides[2]) / 2;

        square_area = sqrt((triangle_perim * (triangle_perim - triangle_sides[0])
                                           * (triangle_perim - triangle_sides[1])
                                           * (triangle_perim - triangle_sides[2]))) * 2;
        img_square_area = bin_barcode.cols * bin_barcode.rows;

        if (square_area > (img_square_area * 0.2))
        {
            suare_flag = true;
        }
    }
    else
    {
        local_points_flag = true;
    }
    if ((suare_flag || local_points_flag) && purpose == SHRINKING)
    {
        localization_points.clear();
        bin_barcode = resized_bin_barcode.clone();
        list_lines_x = searchHorizontalLines();
        if( list_lines_x.empty() ) { return false; }
        list_lines_y = separateVerticalLines(list_lines_x);
        if( list_lines_y.empty() ) { return false; }

        kmeans(list_lines_y, 3, labels,
               TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
               3, KMEANS_PP_CENTERS, localization_points);

        fixationPoints(localization_points);
        if (localization_points.size() != 3) { return false; }

        const int width  = cvRound(bin_barcode.size().width  * coeff_expansion);
        const int height = cvRound(bin_barcode.size().height * coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
        for (size_t i = 0; i < localization_points.size(); i++)
        {
            localization_points[i] *= coeff_expansion;
        }
    }
    if (purpose == ZOOMING)
    {
        const int width  = cvRound(bin_barcode.size().width  / coeff_expansion);
        const int height = cvRound(bin_barcode.size().height / coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
        for (size_t i = 0; i < localization_points.size(); i++)
        {
            localization_points[i] /= coeff_expansion;
        }
    }

    for (size_t i = 0; i < localization_points.size(); i++)
    {
        for (size_t j = i + 1; j < localization_points.size(); j++)
        {
            if (norm(localization_points[i] - localization_points[j]) < 10)
            {
                return false;
            }
        }
    }

    return true;

}

bool QRDetect::computeTransformationPoints()
{
    CV_TRACE_FUNCTION();
    if (localization_points.size() != 3) { return false; }

    vector<Point> locations, non_zero_elem[3], newHull;
    vector<Point2f> new_non_zero_elem[3];
    for (size_t i = 0; i < 3; i++)
    {
        Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
        uint8_t next_pixel, future_pixel = 255;
        int count_test_lines = 0, index = cvRound(localization_points[i].x);
        for (; index < bin_barcode.cols - 1; index++)
        {
            next_pixel = bin_barcode.ptr<uint8_t>(cvRound(localization_points[i].y))[index + 1];
            if (next_pixel == future_pixel)
            {
                future_pixel = static_cast<uint8_t>(~future_pixel);
                count_test_lines++;
                if (count_test_lines == 2)
                {
                    floodFill(bin_barcode, mask,
                              Point(index + 1, cvRound(localization_points[i].y)), 255,
                              0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
                    break;
                }
            }
        }
        Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));
        findNonZero(mask_roi, non_zero_elem[i]);
        newHull.insert(newHull.end(), non_zero_elem[i].begin(), non_zero_elem[i].end());
    }
    convexHull(newHull, locations);
    for (size_t i = 0; i < locations.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            for (size_t k = 0; k < non_zero_elem[j].size(); k++)
            {
                if (locations[i] == non_zero_elem[j][k])
                {
                    new_non_zero_elem[j].push_back(locations[i]);
                }
            }
        }
    }

    double pentagon_diag_norm = -1;
    Point2f down_left_edge_point, up_right_edge_point, up_left_edge_point;
    for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
    {
        for (size_t j = 0; j < new_non_zero_elem[2].size(); j++)
        {
            double temp_norm = norm(new_non_zero_elem[1][i] - new_non_zero_elem[2][j]);
            if (temp_norm > pentagon_diag_norm)
            {
                down_left_edge_point = new_non_zero_elem[1][i];
                up_right_edge_point  = new_non_zero_elem[2][j];
                pentagon_diag_norm = temp_norm;
            }
        }
    }

    if (down_left_edge_point == Point2f(0, 0) ||
        up_right_edge_point  == Point2f(0, 0) ||
        new_non_zero_elem[0].size() == 0) { return false; }

    double max_area = -1;
    up_left_edge_point = new_non_zero_elem[0][0];

    for (size_t i = 0; i < new_non_zero_elem[0].size(); i++)
    {
        vector<Point2f> list_edge_points;
        list_edge_points.push_back(new_non_zero_elem[0][i]);
        list_edge_points.push_back(down_left_edge_point);
        list_edge_points.push_back(up_right_edge_point);

        double temp_area = fabs(contourArea(list_edge_points));
        if (max_area < temp_area)
        {
            up_left_edge_point = new_non_zero_elem[0][i];
            max_area = temp_area;
        }
    }

    Point2f down_max_delta_point, up_max_delta_point;
    double norm_down_max_delta = -1, norm_up_max_delta = -1;
    for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
    {
        double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[1][i])
                               + norm(down_left_edge_point - new_non_zero_elem[1][i]);
        if (norm_down_max_delta < temp_norm_delta)
        {
            down_max_delta_point = new_non_zero_elem[1][i];
            norm_down_max_delta = temp_norm_delta;
        }
    }


    for (size_t i = 0; i < new_non_zero_elem[2].size(); i++)
    {
        double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[2][i])
                               + norm(up_right_edge_point - new_non_zero_elem[2][i]);
        if (norm_up_max_delta < temp_norm_delta)
        {
            up_max_delta_point = new_non_zero_elem[2][i];
            norm_up_max_delta = temp_norm_delta;
        }
    }

    transformation_points.push_back(down_left_edge_point);
    transformation_points.push_back(up_left_edge_point);
    transformation_points.push_back(up_right_edge_point);
    transformation_points.push_back(
        intersectionLines(down_left_edge_point, down_max_delta_point,
                          up_right_edge_point, up_max_delta_point));

    vector<Point2f> quadrilateral = getQuadrilateral(transformation_points);
    transformation_points = quadrilateral;

    int width = bin_barcode.size().width;
    int height = bin_barcode.size().height;
    for (size_t i = 0; i < transformation_points.size(); i++)
    {
        if ((cvRound(transformation_points[i].x) > width) ||
            (cvRound(transformation_points[i].y) > height)) { return false; }
    }
    return true;
}

Point2f QRDetect::intersectionLines(Point2f a1, Point2f a2, Point2f b1, Point2f b2)
{
    Point2f result_square_angle(
                              ((a1.x * a2.y  -  a1.y * a2.x) * (b1.x - b2.x) -
                               (b1.x * b2.y  -  b1.y * b2.x) * (a1.x - a2.x)) /
                              ((a1.x - a2.x) * (b1.y - b2.y) -
                               (a1.y - a2.y) * (b1.x - b2.x)),
                              ((a1.x * a2.y  -  a1.y * a2.x) * (b1.y - b2.y) -
                               (b1.x * b2.y  -  b1.y * b2.x) * (a1.y - a2.y)) /
                              ((a1.x - a2.x) * (b1.y - b2.y) -
                               (a1.y - a2.y) * (b1.x - b2.x))
                              );
    return result_square_angle;
}

// test function (if true then ------> else <------ )
bool QRDetect::testBypassRoute(vector<Point2f> hull, int start, int finish)
{
    CV_TRACE_FUNCTION();
    int index_hull = start, next_index_hull, hull_size = (int)hull.size();
    double test_length[2] = { 0.0, 0.0 };
    do
    {
        next_index_hull = index_hull + 1;
        if (next_index_hull == hull_size) { next_index_hull = 0; }
        test_length[0] += norm(hull[index_hull] - hull[next_index_hull]);
        index_hull = next_index_hull;
    }
    while(index_hull != finish);

    index_hull = start;
    do
    {
        next_index_hull = index_hull - 1;
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }
        test_length[1] += norm(hull[index_hull] - hull[next_index_hull]);
        index_hull = next_index_hull;
    }
    while(index_hull != finish);

    if (test_length[0] < test_length[1]) { return true; } else { return false; }
}

vector<Point2f> QRDetect::getQuadrilateral(vector<Point2f> angle_list)
{
    CV_TRACE_FUNCTION();
    size_t angle_size = angle_list.size();
    uint8_t value, mask_value;
    Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
    Mat fill_bin_barcode = bin_barcode.clone();
    for (size_t i = 0; i < angle_size; i++)
    {
        LineIterator line_iter(bin_barcode, angle_list[ i      % angle_size],
                                            angle_list[(i + 1) % angle_size]);
        for(int j = 0; j < line_iter.count; j++, ++line_iter)
        {
            Point p = line_iter.pos();
            value = bin_barcode.at<uint8_t>(p);
            mask_value = mask.at<uint8_t>(p + Point(1, 1));
            if (value == 0 && mask_value == 0)
            {
                floodFill(fill_bin_barcode, mask, p, 255,
                          0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
            }
        }
    }
    vector<Point> locations;
    Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));

    findNonZero(mask_roi, locations);

    for (size_t i = 0; i < angle_list.size(); i++)
    {
        int x = cvRound(angle_list[i].x);
        int y = cvRound(angle_list[i].y);
        locations.push_back(Point(x, y));
    }

    vector<Point> integer_hull;
    convexHull(locations, integer_hull);
    int hull_size = (int)integer_hull.size();
    vector<Point2f> hull(hull_size);
    for (int i = 0; i < hull_size; i++)
    {
        float x = saturate_cast<float>(integer_hull[i].x);
        float y = saturate_cast<float>(integer_hull[i].y);
        hull[i] = Point2f(x, y);
    }

    const double experimental_area = fabs(contourArea(hull));

    vector<Point2f> result_hull_point(angle_size);
    double min_norm;
    for (size_t i = 0; i < angle_size; i++)
    {
        min_norm = std::numeric_limits<double>::max();
        Point closest_pnt;
        for (int j = 0; j < hull_size; j++)
        {
            double temp_norm = norm(hull[j] - angle_list[i]);
            if (min_norm > temp_norm)
            {
                min_norm = temp_norm;
                closest_pnt = hull[j];
            }
        }
        result_hull_point[i] = closest_pnt;
    }

    int start_line[2] = { 0, 0 }, finish_line[2] = { 0, 0 }, unstable_pnt = 0;
    for (int i = 0; i < hull_size; i++)
    {
        if (result_hull_point[2] == hull[i]) { start_line[0] = i; }
        if (result_hull_point[1] == hull[i]) { finish_line[0] = start_line[1] = i; }
        if (result_hull_point[0] == hull[i]) { finish_line[1] = i; }
        if (result_hull_point[3] == hull[i]) { unstable_pnt = i; }
    }

    int index_hull, extra_index_hull, next_index_hull, extra_next_index_hull;
    Point result_side_begin[4], result_side_end[4];

    bool bypass_orientation = testBypassRoute(hull, start_line[0], finish_line[0]);

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[index_hull] - angle_list[1]) >
        norm(hull[index_hull] - angle_list[2]) ? angle_list[2] : angle_list[1];

        Point intrsc_line_hull =
        intersectionLines(hull[index_hull], hull[next_index_hull],
                          angle_list[1], angle_list[2]);
        double temp_norm = getCosVectors(hull[index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[index_hull] - hull[next_index_hull]) >
            norm(angle_list[1] - angle_list[2]) * 0.1)
        {
            min_norm = temp_norm;
            result_side_begin[0] = hull[index_hull];
            result_side_end[0]   = hull[next_index_hull];
        }


        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[0]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[0] = angle_list[1];
        result_side_end[0]   = angle_list[2];
    }

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[1];
    bypass_orientation = testBypassRoute(hull, start_line[1], finish_line[1]);
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[index_hull] - angle_list[0]) >
        norm(hull[index_hull] - angle_list[1]) ? angle_list[1] : angle_list[0];

        Point intrsc_line_hull =
        intersectionLines(hull[index_hull], hull[next_index_hull],
                          angle_list[0], angle_list[1]);
        double temp_norm = getCosVectors(hull[index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[index_hull] - hull[next_index_hull]) >
            norm(angle_list[0] - angle_list[1]) * 0.05)
        {
            min_norm = temp_norm;
            result_side_begin[1] = hull[index_hull];
            result_side_end[1]   = hull[next_index_hull];
        }

        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[1]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[1] = angle_list[0];
        result_side_end[1]   = angle_list[1];
    }

    bypass_orientation = testBypassRoute(hull, start_line[0], unstable_pnt);
    const bool extra_bypass_orientation = testBypassRoute(hull, finish_line[1], unstable_pnt);

    vector<Point2f> result_angle_list(4), test_result_angle_list(4);
    double min_diff_area = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    const double standart_norm = std::max(
        norm(result_side_begin[0] - result_side_end[0]),
        norm(result_side_begin[1] - result_side_end[1]));
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        if (norm(hull[index_hull] - hull[next_index_hull]) < standart_norm * 0.1)
        { index_hull = next_index_hull; continue; }

        extra_index_hull = finish_line[1];
        do
        {
            if (extra_bypass_orientation) { extra_next_index_hull = extra_index_hull + 1; }
            else { extra_next_index_hull = extra_index_hull - 1; }

            if (extra_next_index_hull == hull_size) { extra_next_index_hull = 0; }
            if (extra_next_index_hull == -1) { extra_next_index_hull = hull_size - 1; }

            if (norm(hull[extra_index_hull] - hull[extra_next_index_hull]) < standart_norm * 0.1)
            { extra_index_hull = extra_next_index_hull; continue; }

            test_result_angle_list[0]
            = intersectionLines(result_side_begin[0], result_side_end[0],
                                result_side_begin[1], result_side_end[1]);
            test_result_angle_list[1]
            = intersectionLines(result_side_begin[1], result_side_end[1],
                                hull[extra_index_hull], hull[extra_next_index_hull]);
            test_result_angle_list[2]
            = intersectionLines(hull[extra_index_hull], hull[extra_next_index_hull],
                                hull[index_hull], hull[next_index_hull]);
            test_result_angle_list[3]
            = intersectionLines(hull[index_hull], hull[next_index_hull],
                                result_side_begin[0], result_side_end[0]);

            const double test_diff_area
                = fabs(fabs(contourArea(test_result_angle_list)) - experimental_area);
            if (min_diff_area > test_diff_area)
            {
                min_diff_area = test_diff_area;
                for (size_t i = 0; i < test_result_angle_list.size(); i++)
                {
                    result_angle_list[i] = test_result_angle_list[i];
                }
            }

            extra_index_hull = extra_next_index_hull;
        }
        while(extra_index_hull != unstable_pnt);

        index_hull = next_index_hull;
    }
    while(index_hull != unstable_pnt);

    // check label points
    if (norm(result_angle_list[0] - angle_list[1]) > 2) { result_angle_list[0] = angle_list[1]; }
    if (norm(result_angle_list[1] - angle_list[0]) > 2) { result_angle_list[1] = angle_list[0]; }
    if (norm(result_angle_list[3] - angle_list[2]) > 2) { result_angle_list[3] = angle_list[2]; }

    // check calculation point
    if (norm(result_angle_list[2] - angle_list[3]) >
       (norm(result_angle_list[0] - result_angle_list[1]) +
        norm(result_angle_list[0] - result_angle_list[3])) * 0.5 )
    { result_angle_list[2] = angle_list[3]; }

    return result_angle_list;
}

//      / | b
//     /  |
//    /   |
//  a/    | c

inline double QRDetect::getCosVectors(Point2f a, Point2f b, Point2f c)
{
    return ((a - b).x * (c - b).x + (a - b).y * (c - b).y) / (norm(a - b) * norm(c - b));
}

struct QRCodeDetector::Impl
{
public:
    Impl() { epsX = 0.2; epsY = 0.1; }
    ~Impl() {}

    double epsX, epsY;
};

QRCodeDetector::QRCodeDetector() : p(new Impl) {}
QRCodeDetector::~QRCodeDetector() {}

void QRCodeDetector::setEpsX(double epsX) { p->epsX = epsX; }
void QRCodeDetector::setEpsY(double epsY) { p->epsY = epsY; }

bool QRCodeDetector::detect(InputArray in, OutputArray points) const
{
    Mat inarr;
    if (!checkQRInputImage(in, inarr))
        return false;

    QRDetect qrdet;
    qrdet.init(inarr, p->epsX, p->epsY);
    if (!qrdet.localization()) { return false; }
    if (!qrdet.computeTransformationPoints()) { return false; }
    vector<Point2f> pnts2f = qrdet.getTransformationPoints();
    updatePointsResult(points, pnts2f);
    return true;
}

class QRDecode
{
public:
    QRDecode();
    void init(const Mat &src, const vector<Point2f> &points);
    Mat getIntermediateBarcode() { return intermediate; }
    Mat getStraightBarcode() { return straight; }
    size_t getVersion() { return version_level; }
    std::string getDecodeInformation() { return result_info; }
    bool fullDecodingProcess();

protected:
    bool updatePerspective();
    bool versionDefinition();
    bool samplingForVersion();
    bool decodingProcess();

    bool readAndCorrectFormat(uint16_t& format);
    bool correctFormat(uint16_t& format);

    void  unmaskData();
    void  readData();
    void  readBit(int x, int y,int& count);
    void  rearrangeBlocks();

    bool  correctSingleBlock(int num , int head,Mat & corrected);

    bool decodeCurrentStream();
    bool numericDecoding(int &index);
    bool kanjiDecoding(int &index);
    bool byteDecoding( int &index);
    bool alphaDecoding( int &index);
    bool eciDecoding(int &index);
    bool structureDecoding(int &index);

    bool FNC1FirstDecoding(const std::string & fnc_buffer);
    bool FNC1SecondDecoding(const std::string & fnc_buffer);

    void loadFromBuffer(const std::string & fnc_buffer, const struct DataOfAI &data,size_t & cur_pos,bool &AI_over);

    int remainingBitsCount(const int &index);

    void convertToUTF8(char* src,const char * fromcode );

    Mat original, no_border_intermediate, intermediate, straight;
    Mat unmasked_data;
    vector<Point2f> original_points;

    /**basic information */
    const VersionInfo *version_info ;
    /**principles  about group and blocks*/
    const  BlockParams *cur_ecc_params;

    vector<uint8_t> orignal_data;
    vector<uint8_t> rearranged_data;

    vector<uint8_t> final_data;

    int			version_level;
    int			ecc_level;
    int			mask_type;

    vector<uint8_t>	cur_str;
    int			cur_str_len;

    uint32_t		eci;
    uint32_t      fnc1_second_AI;
    bool fnc1_first;
    bool fnc1_second;

    std::string fnc1_AI;

    std::string result_info;
    uint8_t version_size;
    float test_perspective_size;

};
QRDecode::QRDecode(){
    orignal_data.reserve(max_payload_len);
    rearranged_data.reserve(max_payload_len);
    cur_str.reserve(max_payload_len);
    for(int i = 0 ; i < max_payload_len ;i++){
        orignal_data.push_back(0);
        rearranged_data.push_back(0);
        cur_str.push_back(0);
    }
    fnc1_AI = "";
    cur_str_len=0;
    eci = 0;
    fnc1_first = 0;
    fnc1_second = 0;
}
/*convertToUTF8
 * params @ src(the original info ) fromcode(the original coding mode)
 * func   @ convert from $(fromcode) to utf-8 and update the cur_str and cur_str_len
 * return @
 * */


/*
 * params @ format(uint16_t for returning the format bits) which(select from two different place)
 * return @ can be correct or not
 */

bool QRDecode::readAndCorrectFormat(uint16_t& format){
    /*version_level<=6*/
    uint16_t my_format = 0;
    int my_size = (int)version_size;
    Mat mat_format(1,max_format_length,CV_8UC1,Scalar(0));

    /*read from the left-bottom and upper-right */
    const int xs[2][max_format_length] = {{8, 8, 8, 8, 8, 8, 8, 8,my_size-7,my_size-6,my_size-5,my_size-4,my_size-3,my_size-2,my_size-1},
                                      {0, 1, 2, 3, 4, 5, 7,8,8     ,8     ,8     ,8     ,8     ,8     ,8}};
    const int ys[2][max_format_length] = {{my_size-1,my_size-2,my_size-3,my_size-4,my_size-5,my_size-6,my_size-7,my_size-8,8, 8, 8, 8, 8, 8, 8 },
                                      {8,   8, 8, 8, 8, 8, 8,     8,     7,     5,     4,     3,     2,     1,     0}} ;
    int read_round = 0;
    bool err;

    for(read_round = 0 ; read_round < 2 ; read_round ++ ){
        for (int i = 0; i <max_format_length; i++) {
            uint8_t value=(straight.ptr<uint8_t>(ys[read_round][i])[xs[read_round][i]]==0);
            my_format = my_format*2 + value ;
        }
        err = correctFormat(my_format);
        if(!err){
            my_format = 0 ;
            continue;
        }
        else
            break;
    }
    if(read_round == 2 && !err)
        return err;
    else{
        format=my_format;
        return true;
    }
}

/*correctFormat:
 *  error correct
 *  params @ uint16_t *f_ret,const Mat& my_f_ret(my format info)
 *  return @
 */
bool QRDecode:: correctFormat(uint16_t& format)
{
    /*ori: 110101100100011*/
    /*adjust several bits to check the correcting ability*/
    format^=4;format^=8;
    /*my method: using BCD ways*/
    int format_index=hammingDetect(format);

    if (format_index==-1){
        return false;
    }

    format=after_mask_format[format_index]^0x5412;

    return true;
}


/**readBit:
 *  params @ (x,y)
 *  func @ read from (x,y) as the bitpos^th bit of the  bytepos^th codeword
 *  return @
 */
void QRDecode::readBit(int x, int y, int& count){
    /**judge the reserved area*/
    if (unmasked_data.ptr(y)[x]==invalid_region_value)
        return ;

    /**the bitpos^th bit of the  bytepos^th codeword*/
    int bytepos = count >> 3;/**equal to count/8 */
    int bitpos  = count & 7 ;/**equal to count%8 */

    int v = (unmasked_data.ptr(y)[x]==0);

    /**first read,first lead*/
    if (v){
        orignal_data[bytepos] |= (0x80 >> bitpos);
    }
    count++;
}

/* exponentiation operator
 * params @  x , power
 * return x^power
 * EXP:
 *     (a^n)^x =a^(x+n)
 * */
uint8_t gfPow(uint8_t x , int power){
    return gf_exp[(gf_log[x] * power) % 255];
}

uint8_t gfInverse(uint8_t x){
    return gf_exp[255 - gf_log[x]];
}

/*multiplication in GF
 * params @ x , y
 * return x * y
 * EXP:
 *     a^x * a^y =a^(x+y)
 */
uint8_t gfMul(const uint8_t &x,const uint8_t& y){
    if (x==0 || y==0)
        return 0;
    return gf_exp[(gf_log[x] + gf_log[y])%255];
}

/*division in GF
 * params @ x , y
 * return x / y
 * EXP:
 *     a^x / a^y =a^(x-y)=a^(x+255-y)
 */
uint8_t gfDiv(const uint8_t &x,const uint8_t& y) {
    if (x == 0)
        return 0;
    return gf_exp[(255 - gf_log[y] + gf_log[x]) % 255];
}

/* gfPolyEvaluate :
 *      evaluate a polynomial at a particular value of x, producing a scalar result
 * params @ poly 15bit format_Info,  uint8_t x(a scalar)
 * return @ result

 * using the Horner's method here:
 * 01 x4 + 0f x3 + 36 x2 + 78 x + 40 = (((01 x + 0f) x + 36) x + 78) x + 40
 * doing this by simple addition and multiplication
 * */
uint8_t gfPolyEvaluate(const Mat& poly,uint8_t x){
    /*Note the calculation begins at the high times of items,
         * That's to say , start from the large index in Mat
         * */
    int index=poly.cols-1;
    uint8_t y=poly.ptr(0)[index];
    for(int i =index-1;i>=0;i--){
        y = gfMul(x,y) ^ poly.ptr(0)[i];
    }
    return y;
}
/*
 * func @  multiply a polynomial by a scalar
 * */
Mat gfPolyScaling(const Mat & p,int scalar) {
    int len = p.cols;
    Mat r(1,len,CV_8UC1,Scalar(0));

    for(int i = 0; i < len;i++){
        r.ptr(0)[i] = gfMul(p.ptr(0)[i], (uint8_t)scalar);
    }
    return r;
}
/*
 * func @  "adds" two polynomials (using exclusive-or, as usual).
 * */
Mat gfPolyAdd(const Mat & p,const Mat & q){
    int p_len=p.cols;
    int q_len=q.cols;
    Mat r (1,max(p_len,q_len),CV_8UC1,Scalar(0));
    for (int i = 0; i< p_len ;i++){
        r.ptr(0)[i] = p.ptr(0)[i];
    }
    for (int i = 0; i< q_len ;i++){
        r.ptr(0)[i] ^= q.ptr(0)[i];
    }
    return r;
}

/* multiplication between two polys
 * params @ poly p , poly q
 * return @ result poly = p * q
 */
/*
       10001001
    *  00101010
 ---------------
      10001001
^   10001001
^ 10001001
 ---------------
  1010001111010*/
Mat gfPolyMul(const Mat &p,const Mat &q){
    /* multiplication == addition among items*/
    Mat r(1,p.cols+q.cols-1,CV_8UC1,Scalar(0));
    int len_p=p.cols;
    int len_q=q.cols;
    for(int j = 0; j<len_q;j++) {
        if(!q.ptr(0)[j])
            continue;
        for (int i = 0; i < len_p; i++) {
            if(!p.ptr(0)[i])
                continue;
            r.ptr(0)[i+j] ^= gfMul(p.ptr(0)[i], q.ptr(0)[j]);
        }
    }
    return r;
}

/*gfPolyDiv:
 *   This function is for getting the ECC for the data string ,which is implemented by doing a poly division.
 * params@ const Mat& dividend,const Mat& divisor
 * return@ ECC code/remainder
 *                             12 da df
 *               -----------------------
 *01 0f 36 78 40 ) 12 34 56 00 00 00 00
 *               ^ 12 ee 2b 23 f4
 *              -------------------------
 *                   da 7d 23 f4 00
 *                 ^ da a2 85 79 84
 *                  ---------------------
 *                      df a6 8d 84 00
 *                    ^ df 91 6b fc d9
 *                    -------------------
 *                         37 e6 78 d9
 */
Mat gfPolyDiv(const Mat& dividend,const Mat& divisor,const int& ecc_num) {
    /* Note that the processing starts from the item with high number of times,
         * so item [total-i] is processed for the i-th round
         * Also before the division started , you need to do a shift to the dividend*/
    int times=dividend.cols-(divisor.cols-1);
    int dividend_len=dividend.cols-1;
    int divisor_len=divisor.cols-1;
    /*Mat.ptr(0)[i] stores the coeffient of the x^i*/
    Mat r=dividend.clone();
    for(int i =0;i<times;i++){
        uint8_t coef=r.ptr(0)[dividend_len-i];
        if(coef!=0){
            for (int j = 0; j < divisor.cols; ++j) {
                if(divisor.ptr(0)[divisor_len-j]!=0){
                    r.ptr(0)[dividend_len-i-j]^=gfMul(divisor.ptr(0)[divisor_len-j], coef);
                }
            }
        }
    }
    Mat ecc=r(Range(0,1),Range(0,ecc_num)).clone();
    return ecc;
}

Mat polyGenerator(const int & n ){
    Mat result = (Mat_<uint8_t >(1,1)<<1);
    Mat temp =   (Mat_<uint8_t >(1,2)<<1,1);

    for(int i = 1; i <= n; i++){
        temp.ptr(0)[0]=gfPow(2,i-1);
        result = gfPolyMul(result,temp);
    }
    return result;
}
/*unmaskData
 *func @  unmask the data and make the pixels in the reserved area Scalar(invalid_region_value)
 */
void QRDecode:: unmaskData(){
    unmasked_data=straight.clone();
    /*get mask pattern according to the format*/

    for(int i= 0;i<version_size;i++){
        for(int j= 0;j<version_size;j++){

            if(unmasked_data.ptr(i)[j]==invalid_region_value)
                continue;

            /*Finder*/
            if ((i < 9 && j < 9)||/* Finder + format: top left */
                (i + 8 >= version_size && j < 9)||/* Finder + format: bottom left */
                (i < 9 && j + 8 >= version_size)||/* Finder + format: top right */
                (i == 6 || j == 6))/* Exclude timing patterns */
            {
                unmasked_data.ptr(i)[j]=invalid_region_value;
            }
                /*version_level information*/
            else if (version_level >= 7) {
                if ((i < 6 && j + 11 >= version_size)||(i + 11 >= version_size && j < 6))
                    unmasked_data.ptr(i)[j]=invalid_region_value;
            }
                /*unmask*/
            else if((mask_type==0&&!((i + j) % 2)) ||
                    (mask_type==1&&!(i % 2)) ||
                    (mask_type==2&&!(j % 3)) ||
                    (mask_type==3 && !((i + j) % 3 )) ||
                    (mask_type==4&&!(((i / 2) + (j / 3)) % 2)) ||
                    (mask_type==5&&!((i * j) % 2 + (i * j) % 3))||
                    (mask_type==6&&!(((i * j) % 2 + (i * j) % 3) % 2))||
                    ((mask_type==7 && !(((i * j) % 3 + (i + j) % 2) % 2)))
                    ){
                unmasked_data.ptr(i)[j]^=255;
            }
        }
    }

    /* Exclude alignment patterns */
    for (int a = 0; a < max_alignment && version_info->alignment_pattern[a]; a++) {
        for (int p = a; p < max_alignment && version_info->alignment_pattern[p]; p++) {
            int x=version_info->alignment_pattern[a];
            int y=version_info->alignment_pattern[p];
            /*the alignment patterns MUST NOT overlap the finder patterns or separators*/
            if(unmasked_data.ptr(x)[y]==invalid_region_value)
                continue;
            for(int i=-2;i<=2;i++)
                for(int j=-2;j<=2;j++)
                    unmasked_data.ptr(x+i)[y+j]=invalid_region_value;
        }
    }

}

/*readData
 * func @ read data from the image into codewords in a zig-zag way
 * */
void QRDecode::readData(){
    int y = version_size - 1;
    int x = version_size - 1;
    int dir = -1;
    int count = 0;
    while (x > 0) {
        if (x == 6)
            x--;
        /*read*/
        readBit( x,  y, count);
        readBit( x-1,  y, count);

        y += dir;
        /*change direction when meets border*/
        if (y < 0 || y >= version_size) {
            dir = -dir;
            x -= 2;
            y += dir;
        }
    }
}

/*calBlockSyndromes
 * params @ const Mat & block(current block),
 *          int synd_num(the num of the syndromes),
 *          uint8_t *synd(syndromes for output)
 * func   @ calculate the syndromes for each block
 * */
int calBlockSyndromes(const Mat & block, int synd_num,vector <uint8_t>& synd){
    int nonzero = 0;
    /*the original method*/
    for (int i = 0; i < synd_num; i++) {
        /*get the syndromes by repalcing the x with pow(2,i) and evaluating the results of the equations*/
        uint8_t tmp =gfPolyEvaluate(block, gfPow(2,i));
        /*print for debug*/
        if (tmp)
            nonzero = 1;
        synd.push_back(tmp);
    }
    return nonzero;
}


/*findErrorLocator
 * params @ synd(the syndromes of current block),
 *          errors_len(the number of the errors),
 * func   @ using berlekamp_massey algorithm to calculate the error_locator
 * return @ findErrorLocator
 * */
Mat findErrorLocator(const vector<uint8_t>&synd,size_t & errors_len){
    /*initialize two arrays b and c ,to be zeros , expcet b0<- 1 c0<- 1*/
    size_t synd_num =synd.size();
    /*err_loc & Sigma*/
    Mat C(1,(int)synd_num,CV_8UC1,Scalar(0));
    /*old_loc */
    Mat B(1,(int)synd_num,CV_8UC1,Scalar(0));//a copy of the last C

    B.ptr(0)[0]=1;
    C.ptr(0)[0]=1;

    uint8_t b=1;//a copy of the last discrepancy delta
    size_t L = 0;//the current number of assumed errors
    int m = 1;//the number of iterations

    for(size_t i = 0; i < synd_num ;i++){
        uint8_t delta = synd[i];
        /*cal discrepancy =Sn+ C1*S(n-1) + ... + CL*S(n-L)*/
        for(size_t j = 1 ; j<= L; j++ ){
            delta ^= gfMul(C.ptr(0)[j], synd[i - j]);
        }
        /*shift = x^m*/
        Mat shift(1,(int)synd_num,CV_8UC1,Scalar(0));
        shift.ptr(0)[m]=1;
        /*scale_coeffi = d/b */
        Mat scale_coeffi = gfPolyScaling(shift,gfMul(delta,gfInverse(b)));

        /*if delta == 0 c is the polynomial */
        if(delta == 0){
            /*assumes that C(x) and L are correct for the moment, increments m, and continues*/
            m++;

        }
            /*If delta !=0 , adjust C(x) so that a recalculation of d would be zero*/

        else if(2 * L <= i){
            /*L is updated and algorithm will update B(x), b, increase L, and reset m = 1*/
            Mat t=C.clone();
            /*C(t)=C(t)-(d/b)*x^m*B(t)    //t stands for the iteration times
                 *    =C(t)-(d/b)*x^m*C(t-1)
                 *delta = Sn+ C1*S(n-1) + ... -(d/b)(Sj+ B1*S(j-1) + ...)
                 */
            C=gfPolyAdd(C,gfPolyMul(B,scale_coeffi));

            B = t.clone();
            b=delta;

            L = i + 1 - L;
            m = 1;
        }

        else{
            /*If L equals the actual number of errors, then during the iteration process,
                 * the discrepancies will become zero before n becomes greater than or equal to 2L.*/
            C = gfPolyAdd(C, gfPolyMul(B,scale_coeffi));
            m++;
        }

    }
    errors_len=L;
    /*L is the length of the minimal LFSR for the stream*/
    return C;
}

/*
     * params @ sigma(error locator) , errors_len(length of sigma -1) , msg_len(length of the block)
     * return @ error index(the index of the error in current block)
     * func   @ use Chien's search to calculate the the roots of error locator poly
     *          and to calculate the index of the error in current block
     * */
vector<int > findErrors(const Mat& sigma,const size_t &errors_len,const int & msg_len){
    vector <int> error_index;
    /*optimize to just check the interesting symbols*/
    for(int i = 0; i < msg_len ; i ++){
        int index=msg_len-i-1;
        /* if a^(n-i) is the error postion ,then a^-(n-i) is the root of the poly Sigma
             * use Chien's search to evaluate the polynomial such that each evaluation only takes constant time
             */
        if(gfPolyEvaluate(sigma,gfInverse(gfPow(2,index)))==0){
            error_index.push_back(index);
        }
    }
    /*print out for debugging*/
    //CV_Assert((int)error_index.size()==errors_len);
    if(error_index.size()!=errors_len)
        error_index.clear();
    return error_index;

}
/*
     * params @ msg_in(the orignial blcok) , synd(syndromes) ,
     *          e_loc_poly(error locator poly) , error_index(the index of the error)
     * return @ the corrected block
     * func   @ use Forney algorithm to correct the errors
     *  ps :
     *  Error location polynomial (short for A(x)) = 1 + sigma{lambda(i) * x(i)}
     *  A(x)'=sigma{i * lambda(i) * x(i-1)}
     * */
Mat errorCorrect(const Mat & msg_in ,const vector<uint8_t>&synd,const Mat & e_loc_poly,const vector<int> &error_index){

    size_t border = synd.size();
    size_t err_len= error_index.size();
    Mat msg_out = msg_in.clone() ;

    Mat syndrome(1,(int)border,CV_8UC1,Scalar(0));
    /*change syndrom to mat from calculation*/
    for(size_t i = 1 ; i < border ; i++){
        syndrome.ptr(0)[i]=synd[i];
    }

    /*First calculate the error evaluator polynomial*/
    Mat Omega= gfPolyMul(syndrome,e_loc_poly);

    /*get rid of the first zero ! */
    Omega = Omega(Range(0,1),Range(1,(int)border));

    /*Second use Forney algorithm to compute the magnitudes*/

    /* 1. calculate formal derivative as the denominator */
    Mat err_location_poly_derivative(1,e_loc_poly.cols,CV_8UC1,Scalar(0));

    /*The operator · represents ordinary multiplication (repeated addition in the finite field)!!!
         * that's why the even items are always zero!*/
    for(size_t i = 1;i <= err_len; i++){
        uint8_t tmp = e_loc_poly.ptr(0)[i];
        err_location_poly_derivative.ptr(0)[i-1]=tmp;
        for(size_t j = 1; j < i ; j++)
            err_location_poly_derivative.ptr(0)[i-1]^=tmp;
    }
    /* 2. calculate Omega as the numerator*/
    for (size_t i = 0; i < err_len; i++) {
        uint8_t xinv = gfInverse(gfPow(2, error_index[i]));
        uint8_t denominator = gfPolyEvaluate(err_location_poly_derivative,xinv);
        uint8_t numerator = gfPolyEvaluate(Omega, xinv);
        /*divded them to get the magnitude*/
        uint8_t error_magnitude = gfDiv(numerator,denominator);
        msg_out.ptr(0)[error_index[i]]^=error_magnitude;
    }
    return msg_out;
}

/* correctSingleBlock
 * params @ num(the number of the current block)  head(the beginning index of NUM^th block)
 * func   @ correct NUM^th block
 * */
bool  QRDecode::correctSingleBlock(int num , int head,Mat & corrected){
    int cur_length=0;

    int ecc_num=cur_ecc_params->ecc_codewords;

    if(num<cur_ecc_params->num_blocks_in_G1){
        cur_length=cur_ecc_params->data_codewords_in_G1+ecc_num;
    }
    else{
        cur_length=cur_ecc_params->data_codewords_in_G2+ecc_num;
    }
    vector<uint8_t>synd;

    /*get the block for calBlockSyndromes*/
    Mat cur_block(1,cur_length,CV_8UC1,Scalar(0));
    for(int i = 0 ;i<cur_length ; i++) {
        cur_block.ptr(0)[cur_length-1-i]=rearranged_data[head+i];
    }

    corrected=cur_block.clone();

    if (!calBlockSyndromes(cur_block,ecc_num,synd))
        return true;

    size_t errors_len=0;
    Mat sigma=findErrorLocator(synd,errors_len);

    vector <int> error_index = findErrors(sigma,errors_len,cur_length);
    Mat corrected_block = errorCorrect(cur_block ,synd, sigma, error_index);

    /*check once again*/
    if (calBlockSyndromes(corrected_block,ecc_num,synd)){
        return false;
    }

    corrected=corrected_block.clone();

    return true;
}

/*
 * params @
 * func   @ rearrange the interleaved blocks for later codewode correction
 * */
void QRDecode::rearrangeBlocks(){
    int index=0;
    int offset=cur_ecc_params->num_blocks_in_G1+cur_ecc_params->num_blocks_in_G2;

    /**the beginning of ecc*/
    int offset_ecc= cur_ecc_params->data_codewords_in_G1*cur_ecc_params->num_blocks_in_G1
                    +
                    cur_ecc_params->data_codewords_in_G2*cur_ecc_params->num_blocks_in_G2;
    /**total num of blocks*/
    int total_blocks=cur_ecc_params->num_blocks_in_G1+cur_ecc_params->num_blocks_in_G2;

    /**the offset for one more col in G2*/
    int offset_one_more=total_blocks*cur_ecc_params->data_codewords_in_G1;

    int cur_block_head=0;
    /**get block in group1*/
    for(int i =0;i<total_blocks;i++){
        /**get the data codeword*/
        for(int j = 0;j <cur_ecc_params->data_codewords_in_G1;j++){
            rearranged_data[index]=orignal_data[i+j*offset];
            index++;
        }
        /**one more  col in G2*/
        if(i>=cur_ecc_params->num_blocks_in_G1)
            rearranged_data[index++]=orignal_data[offset_one_more+i-cur_ecc_params->num_blocks_in_G1];
        /**get the ecc codeword*/
        for(int j = 0;j <cur_ecc_params->ecc_codewords;j++)
            rearranged_data[index++]=orignal_data[offset_ecc+i+j*offset];

        Mat  corrected;
        bool is_not_err= correctSingleBlock(i,cur_block_head,corrected);
        int border = (i>=cur_ecc_params->num_blocks_in_G1)?cur_ecc_params->data_codewords_in_G2:cur_ecc_params->data_codewords_in_G1;
        int total =border +cur_ecc_params->ecc_codewords;

        std::string s =" " ;
        for(int j = 0 ; j < border*codeword_len; j++){
            int cur_word=j>>3;
            int cur_bit =codeword_len-1-(j&7);
            final_data.push_back((corrected.ptr(0)[total-1-cur_word]>>(cur_bit)) & (1));
        }
        cur_block_head=index;
        if(!is_not_err){
            return;
        }
    }
}


void QRDecode::init(const Mat &src, const vector<Point2f> &points)
{
    CV_TRACE_FUNCTION();
    vector<Point2f> bbox = points;
    original = src.clone();
    intermediate = Mat::zeros(original.size(), CV_8UC1);
    original_points = bbox;
    version_level = 0;
    version_size = 0;
    test_perspective_size = 251;
    result_info = "";
}

bool QRDecode::updatePerspective()
{
    CV_TRACE_FUNCTION();
    const Point2f centerPt = QRDetect::intersectionLines(original_points[0], original_points[2],
                                                         original_points[1], original_points[3]);
    if (cvIsNaN(centerPt.x) || cvIsNaN(centerPt.y))
        return false;

    const Size temporary_size(cvRound(test_perspective_size), cvRound(test_perspective_size));

    vector<Point2f> perspective_points;
    perspective_points.push_back(Point2f(0.f, 0.f));
    perspective_points.push_back(Point2f(test_perspective_size, 0.f));

    perspective_points.push_back(Point2f(test_perspective_size, test_perspective_size));
    perspective_points.push_back(Point2f(0.f, test_perspective_size));

    perspective_points.push_back(Point2f(test_perspective_size * 0.5f, test_perspective_size * 0.5f));

    vector<Point2f> pts = original_points;
    pts.push_back(centerPt);

    Mat H = findHomography(pts, perspective_points);
    Mat bin_original;
    adaptiveThreshold(original, bin_original, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    Mat temp_intermediate;
    warpPerspective(bin_original, temp_intermediate, H, temporary_size, INTER_NEAREST);
    no_border_intermediate = temp_intermediate(Range(1, temp_intermediate.rows), Range(1, temp_intermediate.cols));

    const int border = cvRound(0.1 * test_perspective_size);
    const int borderType = BORDER_CONSTANT;
    copyMakeBorder(no_border_intermediate, intermediate, border, border, border, border, borderType, Scalar(255));
    return true;
}

inline Point computeOffset(const vector<Point>& v)
{
    // compute the width/height of convex hull
    Rect areaBox = boundingRect(v);

    // compute the good offset
    // the box is consisted by 7 steps
    // to pick the middle of the stripe, it needs to be 1/14 of the size
    const int cStep = 7 * 2;
    Point offset = Point(areaBox.width, areaBox.height);
    offset /= cStep;
    return offset;
}

bool QRDecode::versionDefinition()
{
    CV_TRACE_FUNCTION();
    LineIterator line_iter(intermediate, Point2f(0, 0), Point2f(test_perspective_size, test_perspective_size));
    Point black_point = Point(0, 0);
    for(int j = 0; j < line_iter.count; j++, ++line_iter)
    {
        const uint8_t value = intermediate.at<uint8_t>(line_iter.pos());
        if (value == 0)
        {
            black_point = line_iter.pos();
            break;
        }
    }

    Mat mask = Mat::zeros(intermediate.rows + 2, intermediate.cols + 2, CV_8UC1);
    floodFill(intermediate, mask, black_point, 255, 0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);

    vector<Point> locations, non_zero_elem;
    Mat mask_roi = mask(Range(1, intermediate.rows - 1), Range(1, intermediate.cols - 1));
    findNonZero(mask_roi, non_zero_elem);
    convexHull(non_zero_elem, locations);
    Point offset = computeOffset(locations);

    Point temp_remote = locations[0], remote_point;
    const Point delta_diff = offset;
    for (size_t i = 0; i < locations.size(); i++)
    {
        if (norm(black_point - temp_remote) <= norm(black_point - locations[i]))
        {
            const uint8_t value = intermediate.at<uint8_t>(temp_remote - delta_diff);
            temp_remote = locations[i];
            if (value == 0) { remote_point = temp_remote - delta_diff; }
            else { remote_point = temp_remote - (delta_diff / 2); }
        }
    }

    size_t transition_x = 0 , transition_y = 0;

    uint8_t future_pixel = 255;
    const uint8_t *intermediate_row = intermediate.ptr<uint8_t>(remote_point.y);
    for(int i = remote_point.x; i < intermediate.cols; i++)
    {
        if (intermediate_row[i] == future_pixel)
        {
            future_pixel = static_cast<uint8_t>(~future_pixel);
            transition_x++;
        }
    }

    future_pixel = 255;
    for(int j = remote_point.y; j < intermediate.rows; j++)
    {
        const uint8_t value = intermediate.at<uint8_t>(Point(j, remote_point.x));
        if (value == future_pixel)
        {
            future_pixel = static_cast<uint8_t>(~future_pixel);
            transition_y++;
        }
    }
    version_level = saturate_cast<uint8_t>((std::min(transition_x, transition_y) - 1) * 0.25 - 1);
    if ( !(  0 < version_level && version_level <= 40 ) ) { return false; }
    version_size = (uint8_t)(21 + (version_level - 1) * 4);
    return true;
}


bool QRDecode::samplingForVersion()
{
    CV_TRACE_FUNCTION();
    const double multiplyingFactor = (version_level < 3)  ? 1 :
                                     (version_level == 3) ? 1.5 :
                                     version_level * (5 + version_level - 4);
    const Size newFactorSize(
                  cvRound(no_border_intermediate.size().width  * multiplyingFactor),
                  cvRound(no_border_intermediate.size().height * multiplyingFactor));
    Mat postIntermediate(newFactorSize, CV_8UC1);
    resize(no_border_intermediate, postIntermediate, newFactorSize, 0, 0, INTER_AREA);

    const int delta_rows = cvRound((postIntermediate.rows * 1.0) / version_size);
    const int delta_cols = cvRound((postIntermediate.cols * 1.0) / version_size);

    /*采样存入straight*/
    straight = Mat(Size(version_size, version_size), CV_8UC1, Scalar(0));


    vector<double> listFrequencyElem;
    for (int r = 0; r < postIntermediate.rows; r += delta_rows)
    {
        for (int c = 0; c < postIntermediate.cols; c += delta_cols)
        {
            Mat tile = postIntermediate(
                           Range(r, min(r + delta_rows, postIntermediate.rows)),
                           Range(c, min(c + delta_cols, postIntermediate.cols)));
            const double frequencyElem = (countNonZero(tile) * 1.0) / tile.total();
            listFrequencyElem.push_back(frequencyElem);
        }
    }

    double dispersionEFE = std::numeric_limits<double>::max();
    double experimentalFrequencyElem = 0;
    for (double expVal = 0; expVal < 1; expVal+=0.001)
    {
        double testDispersionEFE = 0.0;
        for (size_t i = 0; i < listFrequencyElem.size(); i++)
        {
            testDispersionEFE += (listFrequencyElem[i] - expVal) *
                                 (listFrequencyElem[i] - expVal);
        }
        testDispersionEFE /= (listFrequencyElem.size() - 1);
        if (dispersionEFE > testDispersionEFE)
        {
            dispersionEFE = testDispersionEFE;
            experimentalFrequencyElem = expVal;
        }
    }
    straight = Mat(Size(version_size, version_size), CV_8UC1, Scalar(0));
    for (int r = 0; r < version_size * version_size; r++)
    {
        int i   = r / straight.cols;
        int j   = r % straight.cols;
        straight.ptr<uint8_t>(i)[j] = (listFrequencyElem[r] < experimentalFrequencyElem) ? 0 : 255;
    }
    return true;
}


/* getBits
 * params @ bits(the number of bits you need) ptr(the starting position)
 * func   @ from the postion $PTR to get $BITS bits
 * */
int getBits(const int& bits,const vector<uint8_t>& payload , int &pay_index){
    int result=0;
    for(int i =0 ;i<bits;i++){
        result=result<<1;
        result+=payload[pay_index++];
    }
    return result;
}

/* remainingBitsCount
 * params @ ptr(current bit postion)
 * func   @ calculate the remaining number of bits
 * */
int QRDecode::remainingBitsCount(const int &index)
{
    return ((int)final_data.size()-1 - index);
}
/* numericDecoding
 * params @ ptr(current bit postion)
 * func   @ decode the numerical mode
 * */
bool QRDecode::numericDecoding(int &index){
    int count = 0;

    /*check version_level to update the bit counter*/
    int bits = 10;
    if(version_level>=27)
        bits=14;
    else if(version_level>=10)
        bits=12;

    std::string fnc_buffer = "";
    count = getBits(bits,final_data,index);

    if (cur_str_len + count + 1 > max_payload_len){
        return false;
    }
    /*divided 3 numerical char into a 10bit group*/

    while (count >= 3) {
        int num = getBits(10, final_data,index);
        fnc_buffer += char(num / 100 + '0');
        fnc_buffer += char((num % 100) / 10 + '0');
        fnc_buffer += char(num % 10 + '0');
        count -= 3;
    }
    /*the final group*/
    if(count == 2){
        /*7 bit group*/
        int num = getBits(7,final_data,index);
        fnc_buffer += char((num % 100) / 10 + '0');
        fnc_buffer += char(num % 10 + '0');
    } else if (count == 1) {
        /*4 bit group*/
        int num = getBits(4, final_data,index);
        fnc_buffer += char(num % 10 + '0');
    }

    if(!fnc1_first) {
        for(size_t i = 0 ; i < fnc_buffer.length();i++){
            cur_str[cur_str_len++] = uint8_t(fnc_buffer[i]);
        }
    }
    else{
        size_t cur_pos = 0;
        while(1){
            fnc1_AI=fnc_buffer[cur_pos++];
            fnc1_AI+=fnc_buffer[cur_pos++];
            bool is_find = false;
            bool AI_over = false;
            int AI_index = 0;
            /*Find 2 bits AI_name*/
            findAIofFNC1(fnc1_AI,AI_index,is_find);
            if(!is_find){
                /*Find 3 bits AI_name*/
                fnc1_AI+=fnc_buffer[cur_pos++];
                findAIofFNC1(fnc1_AI,AI_index,is_find);
                if(!is_find){
                    /*Find 3 bits AI_name*/
                    fnc1_AI+=fnc_buffer[cur_pos++];
                    findAIofFNC1(fnc1_AI,AI_index,is_find);
                }
            }
            if(!is_find){
                return false;
            }
            /*Find AI_name in the table*/
            const struct AIinGS1 *cur_AI = &GS1_AI_database[AI_index];
            /*load in the text*/
            for(size_t i = 0 ; i < cur_AI->data_title.length();i++){
                cur_str[cur_str_len++] = uint8_t(cur_AI->data_title[i]);
            }
            cur_str[cur_str_len++] = uint8_t(':');
            for(int i = 0 ; i < cur_AI->data[0].Data_len;i++){
                if(!cur_AI->data[0].fixed_len){
                    if(fnc_buffer[cur_pos++]=='%'){
                        AI_over = true;
                        break;
                    }
                }
                cur_str[cur_str_len++] = uint8_t(fnc_buffer[cur_pos++]);
            }
            if(AI_over == true)
                continue;

            for(int i = 0 ; i < cur_AI->data[1].Data_len;i++){
                if(!cur_AI->data[1].fixed_len){
                    if(fnc_buffer[cur_pos++]=='%'){
                        AI_over = true;
                        break;
                    }
                }
                cur_str[cur_str_len++] = uint8_t(fnc_buffer[cur_pos++]);
            }
            if(cur_pos >= fnc_buffer.length()){
                break;
            }
        }
    }
    return true;

}
/* byteDecoding
 * params @ ptr(current bit postion)
 * func   @ decode the byte mode
 */
bool QRDecode::byteDecoding(int &index){
    int bits = 8;
    int count = 0;
    /*check version_level to update the bit counter*/
    if(version_level>9)
        bits=16;
    std::string fnc_buffer = "";
    count = getBits(bits,final_data,index);

    if (cur_str_len + count + 1 > max_payload_len){
        return false;
    }

    if (remainingBitsCount(index) < count * 8){
        return false;
    }

    const char* fromcode = getSrcMode(eci);
    for (int i = 0; i < count; i++){
        int tmp =getBits(8,final_data,index);
        if(!strcmp(fromcode , "UTF−8")){
            fnc_buffer+=char(tmp);
        }
        else{
            /**to be continue*/
            //char src_shift_jis[3]={char(tmp)};
            //convertToUTF8(src_shift_jis,fromcode);
        }
    }
    if(fnc1_first){
        if(cur_str_len == 0)
            loadString(cur_str,cur_str_len,"]Q3");
        for (size_t i = 0; i < fnc_buffer.length(); i++) {
            cur_str[cur_str_len++] = uint8_t(fnc_buffer[i]);
        }
    }
    else if(fnc1_second){//
        if(cur_str_len == 0)
            loadString(cur_str,cur_str_len,"]Q5");
        for (size_t i = 0; i < fnc_buffer.length(); i++) {
            cur_str[cur_str_len++] = uint8_t(fnc_buffer[i]);
        }
    }
    else {
        for (size_t i = 0; i < fnc_buffer.length(); i++) {
            cur_str[cur_str_len++] = uint8_t(fnc_buffer[i]);
        }
    }


    return true;
}


/* kanjiDecoding
 * params @ ptr(current bit postion)
 * func   @ decode the KANJI mode
 *
 * Assume X = Shift JIS value , Y = 13bits value
 * P = X -8140 or X - C140
 * L = P % 16^2   H = P / 16^2
 * H*C0 + L = Y
 * H + L/C0 = Y/C0 ~ H ( L>=C0 )
 * L % C0 = Y % C0 ~ _ _ + + + + + +  (when L <0xCO ,L = Y%0xCO
 */
bool QRDecode::kanjiDecoding(int &index){

    const int per_char_len = 13;
    /*initialize the count indicator*/
    int counter = 12;
    if(version_level<10)
        counter = 8;
    else if(version_level < 27)
        counter = 10;

    /*initialize the count length*/
    int count = 0;
    count = getBits(counter,final_data,index);

    /*one char = two byte   one char = 13 bits*/
    if (cur_str_len + count * 2 + 1 >  max_payload_len || remainingBitsCount(index) < count * per_char_len){
        return false;
    }

    /*correction for L_mod_C0*/
    const int addition[2] = {0b00000000 , 0b11000000};
    for (int i = 0; i < count; i++){
        /*Get My bits*/
        int Y =getBits(per_char_len,final_data,index);
        int L_mod_C0 = Y % 0xc0; /*the real L is L_mod_C0 + addition*/
        int H_around = Y / 0xc0; /*the real H is H_around + (L>=C0)*/
        /*the real L and H */
        int L = 0; int H = 0 ;
        bool is_err = true ;
        /*correction for L_mod */
        for(int j = 0 ; j < 2 ; j++){
            L = addition[j] + L_mod_C0 ;
            H = H_around - (L>=0xc0);
            /*check if is equal to the original bits*/
            if(Y == H*0xc0+L){
                is_err = false;
                break;
            }
        }

        if(is_err){
            return false;
        }
        /*get the subtract value */
        uint16_t subtract =uint16_t((H<<8) + L);
        uint16_t result = 0;
        if (0xe040-0xc140 <= subtract && subtract <= 0xebbf - 0xc140) {
            result = subtract + 0xc140;
        } else {
            result = subtract + 0x8140;
        }

        cur_str[cur_str_len++] = result >> 8;
        cur_str[cur_str_len++] = result & 0xff;
    }
    return true;
}


/* alphaDecoding
* params @ ptr(current bit postion)
* func   @ decode the alpha mode
* */
bool QRDecode::alphaDecoding(int &index){
    /*alpha table*/
    static const char *alpha_map =
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

    std::string fnc_buffer = "";

    int count = 0;
    /*initialize the count indicator*/
    int counter = 13;
    if(version_level<10)
        counter = 9;
    else if(version_level < 27)
        counter = 11;
    /*string length*/
    count = getBits(counter,final_data,index);

    if (cur_str_len + count + 1 > max_payload_len){
        return false;
    }
    /*11bits at a time */
    while (count >= 2) {
        if(remainingBitsCount(index)<11){
            return false;
        }
        int num = getBits(11,final_data,index);
        /*divided into to parts*/
        int H = num/45;
        int L = num%45;
        fnc_buffer+=alpha_map[H];
        fnc_buffer+=alpha_map[L];
        count -= 2;
    }
    /*remaining 6 bits*/
    if (count!=0){
        if(remainingBitsCount(index)<6){
            return false;
        }
        int num = getBits(6,final_data,index);
        fnc_buffer+=alpha_map[num];
    }

    if(fnc1_first){
        if(cur_str_len == 0)
            loadString(cur_str,cur_str_len,"]Q3");
        for (size_t i = 0; i < fnc_buffer.length(); i++) {
            cur_str[cur_str_len++] = uint8_t(fnc_buffer[i]);
        }
    }
    else if(fnc1_second){
        if(cur_str_len == 0)
            loadString(cur_str,cur_str_len,"]Q5");
        for (size_t i = 0; i < fnc_buffer.length(); i++) {
            cur_str[cur_str_len++] = uint8_t(fnc_buffer[i]);
        }
    }
    else {
        for (size_t i = 0; i < fnc_buffer.length(); i++) {
            cur_str[cur_str_len++] = uint8_t(fnc_buffer[i]);
        }
    }
    return true;
}
bool QRDecode::structureDecoding(int &index){
    /*two Structured Append codewords follows indicators
     *  The first codeword is symbol sequence indicator
     *  the second one is parity data,
     *      which is identical in all append message to enable all readable symbols are part of the same Structured Append message*/
    if(remainingBitsCount(index)<16){
        return false;
    }
    //int sequence_indicator = getBits(8,final_data,index);
    //int parity_data = getBits(8,final_data,index);
    return true;
}
/* eciDecoding
* params @ ptr(current bit postion)
* func   @ decode the ECI mode
* */
bool QRDecode::eciDecoding(int &index){
    /*ECI Assignment Number is at least 8bits*/
    if (remainingBitsCount(index) < 8){
        return false;
    }

    /*get ECI Assignment Number*/
    eci = (uint32_t)getBits(8,final_data,index);

    /*check the highest two bits*/
    int codeword_value = eci >> 6;

    while(codeword_value > 0){
        if (remainingBitsCount(index) < 8){
            return false;
        }
        eci = (eci << 8) | getBits(8,final_data,index);
        codeword_value /=2;
    }


    return true;
}
/*
 * findAIofFNC1
 * params @ (const std::string & fnc1_AI(input string ),
 *           int & index(matched index in the table),
 *           bool & is_find(find or not))
 * func   @ look up the AI_name in the table
 *
 * */
void findAIofFNC1(const std::string & fnc1_AI, int & index,bool & is_find){
    for(index = 0 ; index < 999 ; index++){
        if(fnc1_AI == GS1_AI_database[index].AI_name){
            is_find = true;
            break;
        }
    }
}
/* loadFromBuffer
 * params @ const std::string & fnc_buffer, (read form buffer)
 *          const struct DataOfAI &data,     (data len struct)
 *          int & cur_pos,bool &AI_over     (have meet the GS(%) or not )
 * func   @ read from the buffer to  cur_str
 * */
void QRDecode::loadFromBuffer(const std::string & fnc_buffer, const struct DataOfAI &data,size_t & cur_pos,bool &AI_over){
    for(int i = 0 ; i <data.Data_len;i++){
        if(!data.fixed_len){
            /*The length is not fixed*/
            if(fnc_buffer[cur_pos]=='%'){
                /*meet the GS */
                AI_over = true;
                break;
            }
            else if(cur_pos>=fnc_buffer.length())
                break;
        }
        /*read*/
        cur_str[cur_str_len++] = uint8_t(fnc_buffer[cur_pos++]);
    }
    return;
}
/* FNC1FirstDecoding
 * params @ fnc_buffer
 * func   @ decode the mode of fnc1_first
 * */
bool QRDecode::FNC1FirstDecoding(const std::string & fnc_buffer){
    size_t cur_pos = 0;
    while(1){
        bool is_find = false;
        bool AI_over = false;
        int index = 0;
        /*First get 2 bits*/
        fnc1_AI=fnc_buffer[cur_pos++];
        fnc1_AI+=fnc_buffer[cur_pos++];
        /*Find 2 bits AI_name*/
        findAIofFNC1(fnc1_AI, index,is_find);
        if(!is_find){
            /*Find 3 bits AI_name*/
            fnc1_AI+=fnc_buffer[cur_pos++];
            findAIofFNC1(fnc1_AI, index,is_find);
            if(!is_find){
                /*Find 3 bits AI_name*/
                fnc1_AI+=fnc_buffer[cur_pos++];
                findAIofFNC1(fnc1_AI, index,is_find);
            }
        }
        if(!is_find){
            return false;
        }
        /*Find AI_name in the table*/
        const struct AIinGS1 *cur_AI = &GS1_AI_database[index];
        /*load in the text*/
        for(size_t i = 0 ; i < cur_AI->data_title.length();i++){
            cur_str[cur_str_len++] = uint8_t(cur_AI->data_title[i]);
        }
        cur_str[cur_str_len++] = uint8_t(':');
        /*load from the buffer*/
        loadFromBuffer(fnc_buffer,cur_AI->data[0],cur_pos,AI_over);
        if(AI_over){
            cur_pos++;
            continue;
        }
        loadFromBuffer(fnc_buffer,cur_AI->data[1],cur_pos,AI_over);
        if(cur_pos >= fnc_buffer.length()){
            break;
        }
    }
    return true;
}

bool QRDecode::FNC1SecondDecoding(const std::string & fnc_buffer){
    size_t cur_pos = 0;
    while(1){
        bool is_find = false;
        bool AI_over = false;
        int index = 0;
        /*First get 2 bits*/
        fnc1_AI=fnc_buffer[cur_pos++];
        fnc1_AI+=fnc_buffer[cur_pos++];
        /*Find 2 bits AI_name*/
        findAIofFNC1(fnc1_AI, index,is_find);

        /*Find AI_name in the table*/
        const struct AIinGS1 *cur_AI = &GS1_AI_database[index];
        /*load in the text*/
        for(size_t i = 0 ; i <cur_AI->data_title.length();i++){
            cur_str[cur_str_len++] = uint8_t(cur_AI->data_title[i]);
        }
        cur_str[cur_str_len++] = uint8_t(':');
        /*load from the buffer*/
        loadFromBuffer(fnc_buffer,cur_AI->data[0],cur_pos,AI_over);
        if(AI_over){
            cur_pos++;
            continue;
        }
        loadFromBuffer(fnc_buffer,cur_AI->data[1],cur_pos,AI_over);
        if(cur_pos >= fnc_buffer.length()){
            break;
        }
    }
    return true;
}

bool QRDecode::decodeCurrentStream(){
    bool err = true;
    int index =0;
    /*test for output*/
    eci = UTF_8;
    while(remainingBitsCount(index)>=4){
        int mode=getBits(4,final_data,index);
        /*select the corresponding decode mode */
        switch (mode){
            case QR_MODE_NUL:
                index = (int)final_data.size()-1;
                break;
            case QR_MODE_NUM:
                err = numericDecoding(index);
                break;
            case QR_MODE_ALPHA:
                err = alphaDecoding(index);
                break;
            case QR_MODE_STRUCTURE:
                err = structureDecoding(index);
                break;
            case QR_MODE_BYTE:
                err = byteDecoding(index);
                break;
            case QR_MODE_KANJI:
                err = kanjiDecoding(index);
                break;
            case QR_MODE_ECI:
                err = eciDecoding(index);
                break;
            case QR_MODE_FNC1FIRST:
                fnc1_first = true;
                break;
            case QR_MODE_FNC1SECOND:
                fnc1_second_AI = getBits(8,final_data,index);
                fnc1_second = true;
                break;
        }
        if(!err)
            return false;
    }
    return err;

}
bool QRDecode::decodingProcess()
{
    bool err;
    uint16_t my_format=0;
    if (straight.empty()) { return false; }
    version_size=uint8_t(straight.size().width);

    if ((version_size - 17) % 4){
        return false;
    }

    /*estimated version_level*/
    version_level = (version_size - 17) / 4;

    if (version_level < 1 ||version_level > max_version){
        return false;
    }

    /* Read format information -- try both locations */
    err = readAndCorrectFormat(my_format);
    if(!err)
        return err;

    /*EC level （1-2）+Mask(3-5) + EC for this string( 6-15) */
    /*get rid of the ecc_code*/
    uint8_t fdata = my_format >> 10;
    ecc_level = fdata >> 3;
    mask_type = fdata & 7;

    version_info =&version_info_database[version_level];
    cur_ecc_params = &version_info->ecc[eccCodeToLevel(ecc_level)];

    unmaskData();
    readData();
    rearrangeBlocks();
    err = decodeCurrentStream();

    if (!err) {
        return false;
    }

    for (int i = 0; i < cur_str_len; i++){
        result_info += cur_str[i];
    }
    return true;
}

bool QRDecode::fullDecodingProcess()
{
    if (!updatePerspective())  { return false; }
    if (!versionDefinition())  { return false; }
    if (!samplingForVersion()) { return false; }
    if (!decodingProcess())    { return false; }
    return true;
}

bool decodeQRCode(InputArray in, InputArray points, std::string &decoded_info, OutputArray straight_qrcode)
{
    QRCodeDetector qrcode;
    decoded_info = qrcode.decode(in, points, straight_qrcode);
    return !decoded_info.empty();
}

cv::String QRCodeDetector::decode(InputArray in, InputArray points,
                                  OutputArray straight_qrcode)
{
    Mat inarr;
    if (!checkQRInputImage(in, inarr))
        return std::string();

    vector<Point2f> src_points;
    points.copyTo(src_points);
    CV_Assert(src_points.size() == 4);
    CV_CheckGT(contourArea(src_points), 0.0, "Invalid QR code source points");

    QRDecode qrdec;
    qrdec.init(inarr, src_points);
    bool ok = qrdec.fullDecodingProcess();

    std::string decoded_info = qrdec.getDecodeInformation();

    if (ok && straight_qrcode.needed())
    {
        qrdec.getStraightBarcode().convertTo(straight_qrcode,
                                             straight_qrcode.fixedType() ?
                                             straight_qrcode.type() : CV_32FC2);
    }

    return ok ? decoded_info : std::string();
}

cv::String QRCodeDetector::detectAndDecode(InputArray in,
                                           OutputArray points_,
                                           OutputArray straight_qrcode)
{
    Mat inarr;
    if (!checkQRInputImage(in, inarr))
    {
        points_.release();
        return std::string();
    }

    vector<Point2f> points;
    bool ok = detect(inarr, points);
    if (!ok)
    {
        points_.release();
        return std::string();
    }
    updatePointsResult(points_, points);
    std::string decoded_info = decode(inarr, points, straight_qrcode);
    return decoded_info;
}

bool detectQRCode(InputArray in, vector<Point> &points, double eps_x, double eps_y)
{
    QRCodeDetector qrdetector;
    qrdetector.setEpsX(eps_x);
    qrdetector.setEpsY(eps_y);

    return qrdetector.detect(in, points);
}

class QRDetectMulti : public QRDetect
{
public:
    void init(const Mat& src, double eps_vertical_ = 0.2, double eps_horizontal_ = 0.1);
    bool localization();
    bool computeTransformationPoints(const size_t cur_ind);
    vector< vector < Point2f > > getTransformationPoints() { return transformation_points;}

protected:
    int findNumberLocalizationPoints(vector<Point2f>& tmp_localization_points);
    void findQRCodeContours(vector<Point2f>& tmp_localization_points, vector< vector< Point2f > >& true_points_group, const int& num_qrcodes);
    bool checkSets(vector<vector<Point2f> >& true_points_group, vector<vector<Point2f> >& true_points_group_copy,
                   vector<Point2f>& tmp_localization_points);
    void deleteUsedPoints(vector<vector<Point2f> >& true_points_group, vector<vector<Point2f> >& loc,
                          vector<Point2f>& tmp_localization_points);
    void fixationPoints(vector<Point2f> &local_point);
    bool checkPoints(vector<Point2f> quadrangle_points);
    bool checkPointsInsideQuadrangle(const vector<Point2f>& quadrangle_points);
    bool checkPointsInsideTriangle(const vector<Point2f>& triangle_points);

    Mat bin_barcode_fullsize, bin_barcode_temp;
    vector<Point2f> not_resized_loc_points;
    vector<Point2f> resized_loc_points;
    vector< vector< Point2f > > localization_points, transformation_points;
    struct compareDistanse_y
    {
        bool operator()(const Point2f& a, const Point2f& b) const
        {
            return a.y < b.y;
        }
    };
    struct compareSquare
    {
        const vector<Point2f>& points;
        compareSquare(const vector<Point2f>& points_) : points(points_) {}
        bool operator()(const Vec3i& a, const Vec3i& b) const;
    };
    Mat original;
    class ParallelSearch : public ParallelLoopBody
    {
    public:
        ParallelSearch(vector< vector< Point2f > >& true_points_group_,
                vector< vector< Point2f > >& loc_, int iter_, vector<int>& end_,
                vector< vector< Vec3i > >& all_points_,
                QRDetectMulti& cl_)
        :
            true_points_group(true_points_group_),
            loc(loc_),
            iter(iter_),
            end(end_),
            all_points(all_points_),
            cl(cl_)
        {
        }
        void operator()(const Range& range) const CV_OVERRIDE;
        vector< vector< Point2f > >& true_points_group;
        vector< vector< Point2f > >& loc;
        int iter;
        vector<int>& end;
        vector< vector< Vec3i > >& all_points;
        QRDetectMulti& cl;
    };
};

void QRDetectMulti::ParallelSearch::operator()(const Range& range) const
{
    for (int s = range.start; s < range.end; s++)
    {
        bool flag = false;
        for (int r = iter; r < end[s]; r++)
        {
            if (flag)
                break;

            size_t x = iter + s;
            size_t k = r - iter;
            vector<Point2f> triangle;

            for (int l = 0; l < 3; l++)
            {
                triangle.push_back(true_points_group[s][all_points[s][k][l]]);
            }

            if (cl.checkPointsInsideTriangle(triangle))
            {
                bool flag_for_break = false;
                cl.fixationPoints(triangle);
                if (triangle.size() == 3)
                {
                    cl.localization_points[x] = triangle;
                    if (cl.purpose == cl.SHRINKING)
                    {

                        for (size_t j = 0; j < 3; j++)
                        {
                            cl.localization_points[x][j] *= cl.coeff_expansion;
                        }
                    }
                    else if (cl.purpose == cl.ZOOMING)
                    {
                        for (size_t j = 0; j < 3; j++)
                        {
                            cl.localization_points[x][j] /= cl.coeff_expansion;
                        }
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        for (size_t j = i + 1; j < 3; j++)
                        {
                            if (norm(cl.localization_points[x][i] - cl.localization_points[x][j]) < 10)
                            {
                                cl.localization_points[x].clear();
                                flag_for_break = true;
                                break;
                            }
                        }
                        if (flag_for_break)
                            break;
                    }
                    if ((!flag_for_break)
                            && (cl.localization_points[x].size() == 3)
                            && (cl.computeTransformationPoints(x))
                            && (cl.checkPointsInsideQuadrangle(cl.transformation_points[x]))
                            && (cl.checkPoints(cl.transformation_points[x])))
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            loc[s][all_points[s][k][l]].x = -1;
                        }

                        flag = true;
                        break;
                    }
                }
                if (flag)
                {
                    break;
                }
                else
                {
                    cl.transformation_points[x].clear();
                    cl.localization_points[x].clear();
                }
            }
        }
    }
}

void QRDetectMulti::init(const Mat& src, double eps_vertical_, double eps_horizontal_)
{
    CV_TRACE_FUNCTION();

    CV_Assert(!src.empty());
    const double min_side = std::min(src.size().width, src.size().height);
    if (min_side < 512.0)
    {
        purpose = ZOOMING;
        coeff_expansion = 512.0 / min_side;
        const int width  = cvRound(src.size().width  * coeff_expansion);
        const int height = cvRound(src.size().height  * coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_LINEAR);
    }
    else if (min_side > 512.0)
    {
        purpose = SHRINKING;
        coeff_expansion = min_side / 512.0;
        const int width  = cvRound(src.size().width  / coeff_expansion);
        const int height = cvRound(src.size().height  / coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_AREA);
    }
    else
    {
        purpose = UNCHANGED;
        coeff_expansion = 1.0;
        barcode = src.clone();
    }

    eps_vertical   = eps_vertical_;
    eps_horizontal = eps_horizontal_;
    adaptiveThreshold(barcode, bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    adaptiveThreshold(src, bin_barcode_fullsize, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
}

void QRDetectMulti::fixationPoints(vector<Point2f> &local_point)
{
    CV_TRACE_FUNCTION();

    Point2f v0(local_point[1] - local_point[2]);
    Point2f v1(local_point[0] - local_point[2]);
    Point2f v2(local_point[1] - local_point[0]);

    double cos_angles[3], norm_triangl[3];
    norm_triangl[0] = norm(v0);
    norm_triangl[1] = norm(v1);
    norm_triangl[2] = norm(v2);

    cos_angles[0] = v2.dot(-v1) / (norm_triangl[1] * norm_triangl[2]);
    cos_angles[1] = v2.dot(v0) / (norm_triangl[0] * norm_triangl[2]);
    cos_angles[2] = v1.dot(v0) / (norm_triangl[0] * norm_triangl[1]);

    const double angle_barrier = 0.85;
    if (fabs(cos_angles[0]) > angle_barrier || fabs(cos_angles[1]) > angle_barrier || fabs(cos_angles[2]) > angle_barrier)
    {
        local_point.clear();
        return;
    }

    size_t i_min_cos =
            (cos_angles[0] < cos_angles[1] && cos_angles[0] < cos_angles[2]) ? 0 :
                    (cos_angles[1] < cos_angles[0] && cos_angles[1] < cos_angles[2]) ? 1 : 2;

    size_t index_max = 0;
    double max_area = std::numeric_limits<double>::min();
    for (size_t i = 0; i < local_point.size(); i++)
    {
        const size_t current_index = i % 3;
        const size_t left_index  = (i + 1) % 3;
        const size_t right_index = (i + 2) % 3;

        const Point2f current_point(local_point[current_index]);
        const Point2f left_point(local_point[left_index]);
        const Point2f right_point(local_point[right_index]);
        const Point2f central_point(intersectionLines(
                current_point,
                Point2f(static_cast<float>((local_point[left_index].x + local_point[right_index].x) * 0.5),
                        static_cast<float>((local_point[left_index].y + local_point[right_index].y) * 0.5)),
                Point2f(0, static_cast<float>(bin_barcode_temp.rows - 1)),
                Point2f(static_cast<float>(bin_barcode_temp.cols - 1),
                        static_cast<float>(bin_barcode_temp.rows - 1))));


        vector<Point2f> list_area_pnt;
        list_area_pnt.push_back(current_point);

        vector<LineIterator> list_line_iter;
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, left_point));
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, central_point));
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, right_point));

        for (size_t k = 0; k < list_line_iter.size(); k++)
        {
            LineIterator& li = list_line_iter[k];
            uint8_t future_pixel = 255, count_index = 0;
            for (int j = 0; j < li.count; j++, ++li)
            {
                Point p = li.pos();
                if (p.x >= bin_barcode_temp.cols ||
                    p.y >= bin_barcode_temp.rows)
                {
                    break;
                }

                const uint8_t value = bin_barcode_temp.at<uint8_t>(p);
                if (value == future_pixel)
                {
                    future_pixel = static_cast<uint8_t>(~future_pixel);
                    count_index++;
                    if (count_index == 3)
                    {
                        list_area_pnt.push_back(p);
                        break;
                    }
                }
            }
        }

        const double temp_check_area = contourArea(list_area_pnt);
        if (temp_check_area > max_area)
        {
            index_max = current_index;
            max_area = temp_check_area;
        }

    }
    if (index_max == i_min_cos)
    {
        std::swap(local_point[0], local_point[index_max]);
    }
    else
    {
        local_point.clear();
        return;
    }

    const Point2f rpt = local_point[0], bpt = local_point[1], gpt = local_point[2];
    Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
    if (determinant(m) > 0)
    {
        std::swap(local_point[1], local_point[2]);
    }
}

class BWCounter
{
    size_t white;
    size_t black;
public:
    BWCounter(size_t b = 0, size_t w = 0) : white(w), black(b) {}
    BWCounter& operator+=(const BWCounter& other) { black += other.black; white += other.white; return *this; }
    void count1(uchar pixel) { if (pixel == 255) white++; else if (pixel == 0) black++; }
    double getBWFraction() const { return white == 0 ? std::numeric_limits<double>::infinity() : double(black) / double(white); }
    static BWCounter checkOnePair(const Point2f& tl, const Point2f& tr, const Point2f& bl, const Point2f& br, const Mat& img)
    {
        BWCounter res;
        LineIterator li1(img, tl, tr), li2(img, bl, br);
        for (int i = 0; i < li1.count && i < li2.count; i++, li1++, li2++)
        {
            LineIterator it(img, li1.pos(), li2.pos());
            for (int r = 0; r < it.count; r++, it++)
                res.count1(img.at<uchar>(it.pos()));
        }
        return res;
    }
};

bool QRDetectMulti::checkPoints(vector<Point2f> quadrangle)
{
    if (quadrangle.size() != 4)
        return false;
    std::sort(quadrangle.begin(), quadrangle.end(), compareDistanse_y());
    BWCounter s;
    s += BWCounter::checkOnePair(quadrangle[1], quadrangle[0], quadrangle[2], quadrangle[0], bin_barcode);
    s += BWCounter::checkOnePair(quadrangle[1], quadrangle[3], quadrangle[2], quadrangle[3], bin_barcode);
    const double frac = s.getBWFraction();
    return frac > 0.76 && frac < 1.24;
}

bool QRDetectMulti::checkPointsInsideQuadrangle(const vector<Point2f>& quadrangle_points)
{
    if (quadrangle_points.size() != 4)
        return false;

    int count = 0;
    for (size_t i = 0; i < not_resized_loc_points.size(); i++)
    {
        if (pointPolygonTest(quadrangle_points, not_resized_loc_points[i], true) > 0)
        {
            count++;
        }
    }
    if (count == 3)
        return true;
    else
        return false;
}

bool QRDetectMulti::checkPointsInsideTriangle(const vector<Point2f>& triangle_points)
{
    if (triangle_points.size() != 3)
        return false;
    double eps = 3;
    for (size_t i = 0; i < resized_loc_points.size(); i++)
    {
        if (pointPolygonTest( triangle_points, resized_loc_points[i], true ) > 0)
        {
            if ((abs(resized_loc_points[i].x - triangle_points[0].x) > eps)
                    && (abs(resized_loc_points[i].x - triangle_points[1].x) > eps)
                    && (abs(resized_loc_points[i].x - triangle_points[2].x) > eps))
            {
                return false;
            }
        }
    }
    return true;
}

bool QRDetectMulti::compareSquare::operator()(const Vec3i& a, const Vec3i& b) const
{
    Point2f a0 = points[a[0]];
    Point2f a1 = points[a[1]];
    Point2f a2 = points[a[2]];
    Point2f b0 = points[b[0]];
    Point2f b1 = points[b[1]];
    Point2f b2 = points[b[2]];
    return fabs((a1.x - a0.x) * (a2.y - a0.y) - (a2.x - a0.x) * (a1.y - a0.y)) <
           fabs((b1.x - b0.x) * (b2.y - b0.y) - (b2.x - b0.x) * (b1.y - b0.y));
}

int QRDetectMulti::findNumberLocalizationPoints(vector<Point2f>& tmp_localization_points)
{
    size_t number_possible_purpose = 1;
    if (purpose == SHRINKING)
        number_possible_purpose = 2;
    Mat tmp_shrinking = bin_barcode;
    int tmp_num_points = 0;
    int num_points = -1;
    for (eps_horizontal = 0.1; eps_horizontal < 0.4; eps_horizontal += 0.1)
    {
        tmp_num_points = 0;
        num_points = -1;
        if (purpose == SHRINKING)
            number_possible_purpose = 2;
        else
            number_possible_purpose = 1;
        for (size_t k = 0; k < number_possible_purpose; k++)
        {
            if (k == 1)
                bin_barcode = bin_barcode_fullsize;
            vector<Vec3d> list_lines_x = searchHorizontalLines();
            if (list_lines_x.empty())
            {
                if (k == 0)
                {
                    k = 1;
                    bin_barcode = bin_barcode_fullsize;
                    list_lines_x = searchHorizontalLines();
                    if (list_lines_x.empty())
                        break;
                }
                else
                    break;
            }
            vector<Point2f> list_lines_y = extractVerticalLines(list_lines_x, eps_horizontal);
            if (list_lines_y.size() < 3)
            {
                if (k == 0)
                {
                    k = 1;
                    bin_barcode = bin_barcode_fullsize;
                    list_lines_x = searchHorizontalLines();
                    if (list_lines_x.empty())
                        break;
                    list_lines_y = extractVerticalLines(list_lines_x, eps_horizontal);
                    if (list_lines_y.size() < 3)
                        break;
                }
                else
                    break;
            }
            vector<int> index_list_lines_y;
            for (size_t i = 0; i < list_lines_y.size(); i++)
                index_list_lines_y.push_back(-1);
            num_points = 0;
            for (size_t i = 0; i < list_lines_y.size() - 1; i++)
            {
                for (size_t j = i; j < list_lines_y.size(); j++ )
                {

                    double points_distance = norm(list_lines_y[i] - list_lines_y[j]);
                    if (points_distance <= 10)
                    {
                        if ((index_list_lines_y[i] == -1) && (index_list_lines_y[j] == -1))
                        {
                            index_list_lines_y[i] = num_points;
                            index_list_lines_y[j] = num_points;
                            num_points++;
                        }
                        else if (index_list_lines_y[i] != -1)
                            index_list_lines_y[j] = index_list_lines_y[i];
                        else if (index_list_lines_y[j] != -1)
                            index_list_lines_y[i] = index_list_lines_y[j];
                    }
                }
            }
            for (size_t i = 0; i < index_list_lines_y.size(); i++)
            {
                if (index_list_lines_y[i] == -1)
                {
                    index_list_lines_y[i] = num_points;
                    num_points++;
                }
            }
            if ((tmp_num_points < num_points) && (k == 1))
            {
                purpose = UNCHANGED;
                tmp_num_points = num_points;
                bin_barcode = bin_barcode_fullsize;
                coeff_expansion = 1.0;
            }
            if ((tmp_num_points < num_points) && (k == 0))
            {
                tmp_num_points = num_points;
            }
        }

        if ((tmp_num_points < 3) && (tmp_num_points >= 1))
        {
            const double min_side = std::min(bin_barcode_fullsize.size().width, bin_barcode_fullsize.size().height);
            if (min_side > 512)
            {
                bin_barcode = tmp_shrinking;
                purpose = SHRINKING;
                coeff_expansion = min_side / 512.0;
            }
            if (min_side < 512)
            {
                bin_barcode = tmp_shrinking;
                purpose = ZOOMING;
                coeff_expansion = 512 / min_side;
            }
        }
        else
            break;
    }
    if (purpose == SHRINKING)
        bin_barcode = tmp_shrinking;
    num_points = tmp_num_points;
    vector<Vec3d> list_lines_x = searchHorizontalLines();
    if (list_lines_x.empty())
        return num_points;
    vector<Point2f> list_lines_y = extractVerticalLines(list_lines_x, eps_horizontal);
    if (list_lines_y.size() < 3)
        return num_points;
    if (num_points < 3)
        return num_points;

    Mat labels;
    kmeans(list_lines_y, num_points, labels,
            TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
            num_points, KMEANS_PP_CENTERS, tmp_localization_points);
    bin_barcode_temp = bin_barcode.clone();
    if (purpose == SHRINKING)
    {
        const int width  = cvRound(bin_barcode.size().width  * coeff_expansion);
        const int height = cvRound(bin_barcode.size().height * coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
    }
    else if (purpose == ZOOMING)
    {
        const int width  = cvRound(bin_barcode.size().width  / coeff_expansion);
        const int height = cvRound(bin_barcode.size().height / coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
    }
    else
    {
        bin_barcode = bin_barcode_fullsize.clone();
    }
    return num_points;
}

void QRDetectMulti::findQRCodeContours(vector<Point2f>& tmp_localization_points,
                                      vector< vector< Point2f > >& true_points_group, const int& num_qrcodes)
{
    Mat gray, blur_image, threshold_output;
    Mat bar = barcode;
    const int width  = cvRound(bin_barcode.size().width);
    const int height = cvRound(bin_barcode.size().height);
    Size new_size(width, height);
    resize(bar, bar, new_size, 0, 0, INTER_LINEAR);
    blur(bar, blur_image, Size(3, 3));
    threshold(blur_image, threshold_output, 50, 255, THRESH_BINARY);

    vector< vector< Point > > contours;
    vector<Vec4i> hierarchy;
    findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<Point2f> all_contours_points;
    for (size_t i = 0; i < contours.size(); i++)
    {
        for (size_t j = 0; j < contours[i].size(); j++)
        {
            all_contours_points.push_back(contours[i][j]);
        }
    }
    Mat qrcode_labels;
    vector<Point2f> clustered_localization_points;
    int count_contours = num_qrcodes;
    if (all_contours_points.size() < size_t(num_qrcodes))
        count_contours = (int)all_contours_points.size();
    kmeans(all_contours_points, count_contours, qrcode_labels,
          TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
          count_contours, KMEANS_PP_CENTERS, clustered_localization_points);

    vector< vector< Point2f > > qrcode_clusters(count_contours);
    for (int i = 0; i < count_contours; i++)
        for (int j = 0; j < int(all_contours_points.size()); j++)
        {
            if (qrcode_labels.at<int>(j, 0) == i)
            {
                qrcode_clusters[i].push_back(all_contours_points[j]);
            }
        }
    vector< vector< Point2f > > hull(count_contours);
    for (size_t i = 0; i < qrcode_clusters.size(); i++)
        convexHull(Mat(qrcode_clusters[i]), hull[i]);
    not_resized_loc_points = tmp_localization_points;
    resized_loc_points = tmp_localization_points;
    if (purpose == SHRINKING)
    {
        for (size_t j = 0; j < not_resized_loc_points.size(); j++)
        {
            not_resized_loc_points[j] *= coeff_expansion;
        }
    }
    else if (purpose == ZOOMING)
    {
        for (size_t j = 0; j < not_resized_loc_points.size(); j++)
        {
            not_resized_loc_points[j] /= coeff_expansion;
        }
    }

    true_points_group.resize(hull.size());

    for (size_t j = 0; j < hull.size(); j++)
    {
        for (size_t i = 0; i < not_resized_loc_points.size(); i++)
        {
            if (pointPolygonTest(hull[j], not_resized_loc_points[i], true) > 0)
            {
                true_points_group[j].push_back(tmp_localization_points[i]);
                tmp_localization_points[i].x = -1;
            }

        }
    }
    vector<Point2f> copy;
    for (size_t j = 0; j < tmp_localization_points.size(); j++)
    {
       if (tmp_localization_points[j].x != -1)
            copy.push_back(tmp_localization_points[j]);
    }
    tmp_localization_points = copy;
}

bool QRDetectMulti::checkSets(vector<vector<Point2f> >& true_points_group, vector<vector<Point2f> >& true_points_group_copy,
                              vector<Point2f>& tmp_localization_points)
{
    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        if (true_points_group[i].size() < 3)
        {
            for (size_t j = 0; j < true_points_group[i].size(); j++)
                tmp_localization_points.push_back(true_points_group[i][j]);
            true_points_group[i].clear();
        }
    }
    vector< vector< Point2f > > temp_for_copy;
    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        if (true_points_group[i].size() != 0)
            temp_for_copy.push_back(true_points_group[i]);
    }
    true_points_group = temp_for_copy;
    if (true_points_group.size() == 0)
    {
        true_points_group.push_back(tmp_localization_points);
        tmp_localization_points.clear();
    }
    if (true_points_group.size() == 0)
        return false;
    if (true_points_group[0].size() < 3)
        return false;


    vector<int> set_size(true_points_group.size());
    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        set_size[i] = int( (true_points_group[i].size() - 2 ) * (true_points_group[i].size() - 1) * true_points_group[i].size()) / 6;
    }

    vector< vector< Vec3i > > all_points(true_points_group.size());
    for (size_t i = 0; i < true_points_group.size(); i++)
        all_points[i].resize(set_size[i]);
    int cur_cluster = 0;
    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        cur_cluster = 0;
        for (size_t l = 0; l < true_points_group[i].size() - 2; l++)
            for (size_t j = l + 1; j < true_points_group[i].size() - 1; j++)
                for (size_t k = j + 1; k < true_points_group[i].size(); k++)
                {
                    all_points[i][cur_cluster][0] = int(l);
                    all_points[i][cur_cluster][1] = int(j);
                    all_points[i][cur_cluster][2] = int(k);
                    cur_cluster++;
                }
    }

    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        std::sort(all_points[i].begin(), all_points[i].end(), compareSquare(true_points_group[i]));
    }
    if (true_points_group.size() == 1)
    {
        int check_number = 35;
        if (set_size[0] > check_number)
            set_size[0] = check_number;
        all_points[0].resize(set_size[0]);
    }
    int iter = (int)localization_points.size();
    localization_points.resize(iter + true_points_group.size());
    transformation_points.resize(iter + true_points_group.size());

    true_points_group_copy = true_points_group;
    vector<int> end(true_points_group.size());
    for (size_t i = 0; i < true_points_group.size(); i++)
        end[i] = iter + set_size[i];
    ParallelSearch parallelSearch(true_points_group,
            true_points_group_copy, iter, end, all_points, *this);
    parallel_for_(Range(0, (int)true_points_group.size()), parallelSearch);

    return true;
}

void QRDetectMulti::deleteUsedPoints(vector<vector<Point2f> >& true_points_group, vector<vector<Point2f> >& loc,
                                     vector<Point2f>& tmp_localization_points)
{
    size_t iter = localization_points.size() - true_points_group.size() ;
    for (size_t s = 0; s < true_points_group.size(); s++)
    {
        if (localization_points[iter + s].empty())
            loc[s][0].x = -2;

        if (loc[s].size() == 3)
        {

            if ((true_points_group.size() > 1) || ((true_points_group.size() == 1) && (tmp_localization_points.size() != 0)) )
            {
                for (size_t j = 0; j < true_points_group[s].size(); j++)
                {
                    if (loc[s][j].x != -1)
                    {
                        loc[s][j].x = -1;
                        tmp_localization_points.push_back(true_points_group[s][j]);
                    }
                }
            }
        }
        vector<Point2f> for_copy;
        for (size_t j = 0; j < loc[s].size(); j++)
        {
            if ((loc[s][j].x != -1) && (loc[s][j].x != -2) )
            {
                for_copy.push_back(true_points_group[s][j]);
            }
            if ((loc[s][j].x == -2) && (true_points_group.size() > 1))
            {
                tmp_localization_points.push_back(true_points_group[s][j]);
            }
        }
        true_points_group[s] = for_copy;
    }

    vector< vector< Point2f > > for_copy_loc;
    vector< vector< Point2f > > for_copy_trans;


    for (size_t i = 0; i < localization_points.size(); i++)
    {
        if ((localization_points[i].size() == 3) && (transformation_points[i].size() == 4))
        {
            for_copy_loc.push_back(localization_points[i]);
            for_copy_trans.push_back(transformation_points[i]);
        }
    }
    localization_points = for_copy_loc;
    transformation_points = for_copy_trans;
}

bool QRDetectMulti::localization()
{
    CV_TRACE_FUNCTION();
    vector<Point2f> tmp_localization_points;
    int num_points = findNumberLocalizationPoints(tmp_localization_points);
    if (num_points < 3)
        return false;
    int num_qrcodes = divUp(num_points, 3);
    vector<vector<Point2f> > true_points_group;
    findQRCodeContours(tmp_localization_points, true_points_group, num_qrcodes);
    for (int q = 0; q < num_qrcodes; q++)
    {
       vector<vector<Point2f> > loc;
       size_t iter = localization_points.size();

       if (!checkSets(true_points_group, loc, tmp_localization_points))
            break;
       deleteUsedPoints(true_points_group, loc, tmp_localization_points);
       if ((localization_points.size() - iter) == 1)
           q--;
       if (((localization_points.size() - iter) == 0) && (tmp_localization_points.size() == 0) && (true_points_group.size() == 1) )
            break;
    }
    if ((transformation_points.size() == 0) || (localization_points.size() == 0))
       return false;
    return true;
}

bool QRDetectMulti::computeTransformationPoints(const size_t cur_ind)
{
    CV_TRACE_FUNCTION();

    if (localization_points[cur_ind].size() != 3)
    {
        return false;
    }

    vector<Point> locations, non_zero_elem[3], newHull;
    vector<Point2f> new_non_zero_elem[3];
    for (size_t i = 0; i < 3 ; i++)
    {
        Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
        uint8_t next_pixel, future_pixel = 255;
        int localization_point_x = cvRound(localization_points[cur_ind][i].x);
        int localization_point_y = cvRound(localization_points[cur_ind][i].y);
        int count_test_lines = 0, index = localization_point_x;
        for (; index < bin_barcode.cols - 1; index++)
        {
            next_pixel = bin_barcode.at<uint8_t>(localization_point_y, index + 1);
            if (next_pixel == future_pixel)
            {
                future_pixel = static_cast<uint8_t>(~future_pixel);
                count_test_lines++;

                if (count_test_lines == 2)
                {
                    // TODO avoid drawing functions
                    floodFill(bin_barcode, mask,
                            Point(index + 1, localization_point_y), 255,
                            0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
                    break;
                }
            }

        }
        Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));
        findNonZero(mask_roi, non_zero_elem[i]);
        newHull.insert(newHull.end(), non_zero_elem[i].begin(), non_zero_elem[i].end());
    }
    convexHull(newHull, locations);
    for (size_t i = 0; i < locations.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            for (size_t k = 0; k < non_zero_elem[j].size(); k++)
            {
                if (locations[i] == non_zero_elem[j][k])
                {
                    new_non_zero_elem[j].push_back(locations[i]);
                }
            }
        }
    }

    if (new_non_zero_elem[0].size() == 0)
        return false;

    double pentagon_diag_norm = -1;
    Point2f down_left_edge_point, up_right_edge_point, up_left_edge_point;
    for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
    {
        for (size_t j = 0; j < new_non_zero_elem[2].size(); j++)
        {
            double temp_norm = norm(new_non_zero_elem[1][i] - new_non_zero_elem[2][j]);
            if (temp_norm > pentagon_diag_norm)
            {
                down_left_edge_point = new_non_zero_elem[1][i];
                up_right_edge_point  = new_non_zero_elem[2][j];
                pentagon_diag_norm = temp_norm;
            }
        }
    }

    if (down_left_edge_point == Point2f(0, 0) ||
        up_right_edge_point  == Point2f(0, 0))
    {
        return false;
    }

    double max_area = -1;
    up_left_edge_point = new_non_zero_elem[0][0];

    for (size_t i = 0; i < new_non_zero_elem[0].size(); i++)
    {
        vector<Point2f> list_edge_points;
        list_edge_points.push_back(new_non_zero_elem[0][i]);
        list_edge_points.push_back(down_left_edge_point);
        list_edge_points.push_back(up_right_edge_point);

        double temp_area = fabs(contourArea(list_edge_points));
        if (max_area < temp_area)
        {
            up_left_edge_point = new_non_zero_elem[0][i];
            max_area = temp_area;
        }
    }

    Point2f down_max_delta_point, up_max_delta_point;
    double norm_down_max_delta = -1, norm_up_max_delta = -1;
    for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
    {
        double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[1][i]) + norm(down_left_edge_point - new_non_zero_elem[1][i]);
        if (norm_down_max_delta < temp_norm_delta)
        {
            down_max_delta_point = new_non_zero_elem[1][i];
            norm_down_max_delta = temp_norm_delta;
        }
    }


    for (size_t i = 0; i < new_non_zero_elem[2].size(); i++)
    {
        double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[2][i]) + norm(up_right_edge_point - new_non_zero_elem[2][i]);
        if (norm_up_max_delta < temp_norm_delta)
        {
            up_max_delta_point = new_non_zero_elem[2][i];
            norm_up_max_delta = temp_norm_delta;
        }
    }
    vector<Point2f> tmp_transformation_points;
    tmp_transformation_points.push_back(down_left_edge_point);
    tmp_transformation_points.push_back(up_left_edge_point);
    tmp_transformation_points.push_back(up_right_edge_point);
    tmp_transformation_points.push_back(intersectionLines(
                    down_left_edge_point, down_max_delta_point,
                    up_right_edge_point, up_max_delta_point));
    transformation_points[cur_ind] = tmp_transformation_points;

    vector<Point2f> quadrilateral = getQuadrilateral(transformation_points[cur_ind]);
    transformation_points[cur_ind] = quadrilateral;

    return true;
}

bool QRCodeDetector::detectMulti(InputArray in, OutputArray points) const
{
    Mat inarr;
    if (!checkQRInputImage(in, inarr))
    {
        points.release();
        return false;
    }

    QRDetectMulti qrdet;
    qrdet.init(inarr, p->epsX, p->epsY);
    if (!qrdet.localization())
    {
        points.release();
        return false;
    }
    vector< vector< Point2f > > pnts2f = qrdet.getTransformationPoints();
    vector<Point2f> trans_points;
    for(size_t i = 0; i < pnts2f.size(); i++)
        for(size_t j = 0; j < pnts2f[i].size(); j++)
            trans_points.push_back(pnts2f[i][j]);

    updatePointsResult(points, trans_points);

    return true;
}

class ParallelDecodeProcess : public ParallelLoopBody
{
public:
    ParallelDecodeProcess(Mat& inarr_, vector<QRDecode>& qrdec_, vector<std::string>& decoded_info_,
            vector<Mat>& straight_barcode_, vector< vector< Point2f > >& src_points_)
        : inarr(inarr_), qrdec(qrdec_), decoded_info(decoded_info_)
        , straight_barcode(straight_barcode_), src_points(src_points_)
    {
        // nothing
    }
    void operator()(const Range& range) const CV_OVERRIDE
    {
        for (int i = range.start; i < range.end; i++)
        {
            qrdec[i].init(inarr, src_points[i]);
            bool ok = qrdec[i].fullDecodingProcess();
            if (ok)
            {
                decoded_info[i] = qrdec[i].getDecodeInformation();
                straight_barcode[i] = qrdec[i].getStraightBarcode();
            }
            else if (std::min(inarr.size().width, inarr.size().height) > 512)
            {
                const int min_side = std::min(inarr.size().width, inarr.size().height);
                double coeff_expansion = min_side / 512;
                const int width  = cvRound(inarr.size().width  / coeff_expansion);
                const int height = cvRound(inarr.size().height / coeff_expansion);
                Size new_size(width, height);
                Mat inarr2;
                resize(inarr, inarr2, new_size, 0, 0, INTER_AREA);
                for (size_t j = 0; j < 4; j++)
                {
                    src_points[i][j] /= static_cast<float>(coeff_expansion);
                }
                qrdec[i].init(inarr2, src_points[i]);
                ok = qrdec[i].fullDecodingProcess();
                if (ok)
                {
                    decoded_info[i] = qrdec[i].getDecodeInformation();
                    straight_barcode[i] = qrdec[i].getStraightBarcode();
                }
            }
            if (decoded_info[i].empty())
                decoded_info[i] = "";
        }
    }

private:
    Mat& inarr;
    vector<QRDecode>& qrdec;
    vector<std::string>& decoded_info;
    vector<Mat>& straight_barcode;
    vector< vector< Point2f > >& src_points;

};

bool QRCodeDetector::decodeMulti(
        InputArray img,
        InputArray points,
        CV_OUT std::vector<cv::String>& decoded_info,
        OutputArrayOfArrays straight_qrcode
    ) const
{
    Mat inarr;
    if (!checkQRInputImage(img, inarr))
        return false;
    CV_Assert(points.size().width > 0);
    CV_Assert((points.size().width % 4) == 0);
    vector< vector< Point2f > > src_points ;
    Mat qr_points = points.getMat();
    qr_points = qr_points.reshape(2, 1);
    for (int i = 0; i < qr_points.size().width ; i += 4)
    {
        vector<Point2f> tempMat = qr_points.colRange(i, i + 4);
        if (contourArea(tempMat) > 0.0)
        {
            src_points.push_back(tempMat);
        }
    }
    CV_Assert(src_points.size() > 0);
    vector<QRDecode> qrdec(src_points.size());
    vector<Mat> straight_barcode(src_points.size());
    vector<std::string> info(src_points.size());
    ParallelDecodeProcess parallelDecodeProcess(inarr, qrdec, info, straight_barcode, src_points);
    parallel_for_(Range(0, int(src_points.size())), parallelDecodeProcess);
    vector<Mat> for_copy;
    for (size_t i = 0; i < straight_barcode.size(); i++)
    {
        if (!(straight_barcode[i].empty()))
            for_copy.push_back(straight_barcode[i]);
    }
    straight_barcode = for_copy;
    vector<Mat> tmp_straight_qrcodes;
    if (straight_qrcode.needed())
    {
        for (size_t i = 0; i < straight_barcode.size(); i++)
        {
            Mat tmp_straight_qrcode;
            tmp_straight_qrcodes.push_back(tmp_straight_qrcode);
            straight_barcode[i].convertTo(((OutputArray)tmp_straight_qrcodes[i]),
                                             ((OutputArray)tmp_straight_qrcodes[i]).fixedType() ?
                                             ((OutputArray)tmp_straight_qrcodes[i]).type() : CV_32FC2);
        }
        straight_qrcode.createSameSize(tmp_straight_qrcodes, CV_32FC2);
        straight_qrcode.assign(tmp_straight_qrcodes);
    }
    decoded_info.clear();
    for (size_t i = 0; i < info.size(); i++)
    {
       decoded_info.push_back(info[i]);
    }
    if (!decoded_info.empty())
        return true;
    else
        return false;
}

bool QRCodeDetector::detectAndDecodeMulti(
        InputArray img,
        CV_OUT std::vector<cv::String>& decoded_info,
        OutputArray points_,
        OutputArrayOfArrays straight_qrcode
    ) const
{
    Mat inarr;
    if (!checkQRInputImage(img, inarr))
    {
        points_.release();
        return false;
    }

    vector<Point2f> points;
    bool ok = detectMulti(inarr, points);
    if (!ok)
    {
        points_.release();
        return false;
    }
    updatePointsResult(points_, points);
    decoded_info.clear();
    ok = decodeMulti(inarr, points, decoded_info, straight_qrcode);
    return ok;
}

struct methodBlock{
    int payload_len ;
    vector<uint8_t>	payload;
    int encoding_mode;
    methodBlock();
};

methodBlock::methodBlock() {
    payload_len = encoding_mode = 0;
    payload.reserve(max_payload_len);
}
class QREncoder{
    int			version_level;
    /**pixel width of QR*/
    int version_size ;
    int			ecc_level;
    int			mask_type;
    int         mode_type;
    Mat  format;
    /**the input string */
    std::string input_info;
    /**the original data bits and ecc bits*/
    vector<uint8_t>	payload;
    int			payload_len;
    /**rearranged data bits in encoding style*/
    vector<uint8_t>	rearranged_data;

    Mat original;
    Mat masked_data;

    uint32_t eci;
    //uint32_t fnc1_second_AI;
    bool fnc1_first;
    bool fnc1_second;
    /**basic information */
    const VersionInfo *version_info ;
    /**principles  about group and blocks*/
    const  BlockParams *cur_ecc_params;
public:
    QREncoder(int v,int ecc,int mask,int mode ,std::string input );
    void formatGenerate();

    void encodeByte(std::string input,vector<uint8_t> payload,int &payload_len);
    bool encodeAlpha(std::string input,vector<uint8_t> payload,int &payload_len);
    bool encodeNumeric(std::string input,vector<uint8_t> payload,int &payload_len);
    bool encodeKanji(std::string input,vector<uint8_t> payload,int &payload_len);
    bool encodeAuto(std::string input,vector<uint8_t> payload,int &payload_len);

    bool generateBlock(std::string input, int mode ,methodBlock& block );

    void padBitStream();
    void stringToBits();
    void eccGenerate(vector<Mat>& data_blocks,vector<Mat>& ecc_blocks);
    void rearrangeBlocks(const vector<Mat>& data_blocks,const vector<Mat>& ecc_blocks);
    void maskData();
    void writeReservedArea();
    void writeBit(int x, int y, int& count);
    void writeData();
    void fillReserved();
    void structureFinalMessage();
    Mat QRcodeGenerate();
};
/**
 *
 * */
QREncoder::QREncoder(int v,int ecc,int mask,int mode ,std::string input ){
    version_level = v;
    version_size = (21 + (version_level - 1) * 4);
    ecc_level = ecc;
    mask_type = mask;
    mode_type = mode;
    input_info = input;

    payload.reserve(max_payload_len);
    rearranged_data.reserve(max_payload_len);
    format = Mat(Size(1, 15), CV_8UC1, Scalar(255));
    original = Mat(Size(version_size, version_size), CV_8UC1, Scalar(255));
    masked_data = original.clone();

    payload_len=0;
    eci = 0;
    fnc1_first = 0;
    fnc1_second = 0;

    version_info =&version_info_database[version_level];
    cur_ecc_params = &version_info->ecc[eccCodeToLevel(ecc_level)];
}

/**name  @
 * param @
 * func  @ generate the format bit-stream with EC level and mask type
 */
void QREncoder::formatGenerate(){
    /**poly of format (small index stands for lower items)*/
    Mat Polynomial=Mat(Size(1,max_format_length),CV_8UC1,Scalar(0));
    /** EC level （1-2）+Mask(3-5) + EC for this string( 6-15) */
    std::string Mask_type = decToBin(mask_type,3);
    std::string EC_level = decToBin(ecc_level,2);
    /**ecc coding is a little different*/
    ecc_level = eccLevelToCode(ecc_level);
    std::string version_bits =EC_level + Mask_type;
    /** get shift*/
    Mat binary_bit = (Mat_<uint8_t >(1,5)<<
                                         version_bits[4]-'0',version_bits[3]-'0',version_bits[2]-'0',version_bits[1]-'0',version_bits[0]-'0');//low-->high
    Mat shift = Mat(Size(10,1),CV_8UC1,Scalar(0));
    hconcat(shift,binary_bit,Polynomial);

    /**length of format_generator is 11 not max_format_length*/
    Mat format_generator  = (Mat_<uint8_t >(1,11)<<1,1,1,0,1, 1,0,0,1,0, 1);//low-->high

    /**get ecc by division*/
    Mat ecc_code = gfPolyDiv(Polynomial,format_generator,10);
    hconcat(ecc_code,binary_bit,format);
    /**get masked*/
    Mat mask=(Mat_<uint8_t >(1,max_format_length)<<0,1,0,0,1, 0,0,0,0,0, 1,0,1,0,1);
    for(int i=0;i<max_format_length;i++){
        format.ptr(0)[i]^=mask.ptr(0)[i];
    }
    return ;
}
/**name  @
 * param @
 * func  @ Convert the character string into a bit stream by the encoding of byte mode.
 */
bool QREncoder::encodeAlpha(std::string input,vector<uint8_t> output,int &output_len){
    /**alpha table*/
    std::string alpha_map =
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";
    /**initialize the count indicator*/
    int bits = 13;
    if(version_level<10)
        bits = 9;
    else if(version_level < 27)
        bits = 11;
    /**mode indicator*/
    std::string mode_bits = decToBin(QR_MODE_ALPHA,4);
    loadString(output,output_len,mode_bits, true);
    /**character counter*/
    int str_len = int(input.length());
    std::string counter = decToBin(str_len,bits);
    loadString(output,output_len,counter, true);
    /**encode data*/
    int i;
    for( i = 0 ; i < str_len ; i+=2 ){
        int index_1 = (int)alpha_map.find(input[i]);
        int index_2 = (int)alpha_map.find(input[i+1]);

        if(index_1==-1||index_2==-1){
            output_len = error_mode_occur;
            return false;
        }
        int result = index_1*45 + index_2;

        std::string per_byte = decToBin(result,11);
        loadString(output,output_len,per_byte, true);
    }
    /**last character*/
    if( i %2 !=0 ){
        int index = (int)alpha_map.find(input[i-1]);
        std::string per_byte = decToBin(index,6);
        loadString(output,output_len,per_byte, true);
    }

    return true;
}
/**name  @
 * param @
 * func  @ Convert the character string into a bit stream by the encoding of byte mode.
 */
void QREncoder::encodeByte(std::string input,vector<uint8_t> output,int &output_len){
    int bits = 8;
    /**check version_level to update the bit counter*/
    if(version_level>9)
        bits=16;
    /**mode indicator*/
    std::string mode_bits = decToBin(QR_MODE_BYTE,4);
    loadString(output,output_len,mode_bits, true);
    /**character counter*/
    int str_len = int(input.length());
    std::string counter = decToBin(str_len,bits);
    loadString(output,output_len,counter, true);
    /**encode data*/
    for(int i = 0 ; i < str_len ; i++ ){
        std::string per_byte = decToBin(int(input[i]),8);
        loadString(output,output_len,per_byte, true);
    }
    return ;
}

/** encodeNumeric
 * params@ ptr(current bit postion)
 * func@ decode the numerical mode
 * */
bool QREncoder::encodeNumeric(std::string input,vector<uint8_t> output,int &output_len){
    /*check version_level to update the bit counter*/
    int bits = 10;
    if(version_level>=27)
        bits=14;
    else if(version_level>=10)
        bits=12;

    /**mode indicator*/
    std::string mode_bits = decToBin(QR_MODE_NUM,4);
    loadString(output,output_len,mode_bits, true);
    /**character counter*/
    int str_len = int(input.length());
    std::string counter = decToBin(str_len,bits);
    loadString(output,output_len,counter, true);

    /*divided 3 numerical char into a 10bit group*/
    int count = 0;
    while (count + 3 <= str_len) {
        if(input[count]>'9'||input[count]<'0'||
           input[count+1]>'9'||input[count+1]<'0'||
           input[count+2]>'9'||input[count+2]<'0'){
            output_len = error_mode_occur;
            return false;
        }
        int num = 100*(int)(input[count  ]-'0')+
                   10*(int)(input[count+1]-'0')+
                      (int)(input[count+2]-'0');
        std::string numeric_group = decToBin(num,10);
        loadString(output,output_len,numeric_group, true);
        count += 3;
    }
    /*the final group*/
    if(count + 2 == str_len){
        /*7 bit group*/
        if(input[count]>'9'||input[count]<'0'||
           input[count+1]>'9'||input[count+1]<'0'){
            output_len = error_mode_occur;
            return false;
        }
        int num = 10*(int)(input[count  ]-'0')+
                     (int)(input[count+1]-'0');
        std::string numeric_group = decToBin(num,7);
        loadString(output,output_len,numeric_group, true);

    } else if (count + 1 == str_len) {
        /*4 bit group*/
        if(input[count]>'9'||input[count]<'0'){
            output_len = error_mode_occur;
            return false;
        }
        int num = (int)(input[count  ]-'0');
        std::string numeric_group = decToBin(num,4);
        loadString(output,output_len,numeric_group, true);
    }

    return true;

}

/** encodeKanji
 * params@
 * func@
 * attention@ The input info must be the kanji in SHIFT_JIS encoding set !
 * */
bool QREncoder::encodeKanji(std::string input,vector<uint8_t> output,int &output_len){
    /*initialize the count indicator*/
    int bits = 12;
    if(version_level<10)
        bits = 8;
    else if(version_level < 27)
        bits = 10;

    /**mode indicator*/
    std::string mode_bits = decToBin(QR_MODE_KANJI,4);
    loadString(output,output_len,mode_bits, true);
    /**character counter (is the number of kanji character , not the encoding byte !!!! )*/
    int str_len = int(input.length()) / 2 ;
    std::string counter = decToBin(str_len,bits);
    loadString(output,output_len,counter, true);

    int i = 0 ;
    while(i < str_len*2){
        /**two byte at a time*/
        /**unsigned -> signed be attention!*/
        uint16_t high_byte =(uint16_t)(input[i] & 0xff);
        uint16_t low_byte = (uint16_t)(input[i+1] & 0xff);
        uint16_t per_char = (high_byte<<8) + (low_byte );
        /**subtract*/
        if(0x8140 <= per_char && per_char <= 0x9FFC){
            per_char -= 0x8140;
        }
        else if(0xE040 <= per_char && per_char <= 0xEBBF){
            per_char -= 0xC140;
        }
        /**multiply most significant byte of result by 0xC0*/

        uint16_t new_high = per_char>>8;

        uint16_t result = new_high * 0xC0;
        /**add least significant byte to product from b*/
        result += (per_char & 0xFF);
        /**convert to a 13-bit string */
        std::string char_stream = decToBin(result,13);
        loadString(output,output_len,char_stream, true);

        i+=2;
    }

    return true;
}


bool QREncoder::generateBlock(std::string input , int mode ,struct methodBlock& block ){
    block.payload_len = 0 ;
    block.encoding_mode = mode;
    payload.reserve(max_payload_len);
    bool result = true ;
    switch (mode) {
        case QR_MODE_NUM:
            result = encodeNumeric(input,block.payload,block.payload_len);
            break;
        case QR_MODE_ALPHA:
            result = encodeAlpha(input,block.payload,block.payload_len);
            break;
        case QR_MODE_BYTE:
            encodeByte(input,block.payload,block.payload_len);
            break;
        case QR_MODE_KANJI:
            result = encodeKanji(input,block.payload,block.payload_len);
            break;
    }
    return  result;
}


struct encodingMethods{
    int len;
    vector<methodBlock> blocks;
    encodingMethods(){
        len = 0 ;
        blocks.clear();
    }
    int sum_len(){
        int bits_len = 0 ;
        for(size_t i = 0 ; i < blocks.size(); i++ ){
            bits_len += blocks[i].payload_len;
        }
        return bits_len;
    }
};

/** encodeAuto
 * params@
 * func@ the priority is : numeric -> alphanumeric -> byte -> kanji
 * */
bool QREncoder::encodeAuto(std::string input,vector<uint8_t> output,int &output_len){

    std::string mode_char_set [2] ;
    mode_char_set[0]= "0123456789";///numeric
    mode_char_set[1]= "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";///alphanumeric

    /**DP strategy*/
    vector<encodingMethods> strategy;
    methodBlock last_method;

    encodingMethods head;
    strategy.push_back(head);

    /**no kanji */
    size_t len = input.length();
    std::string cur_string = "";
    for(size_t i = 0 ; i < len ; i++ ){
        cur_string += char (input[i]);
        if(strategy.size() == 1){
            encodingMethods tmp;
            if(!generateBlock(cur_string,QR_MODE_NUM,last_method)){
                if(!generateBlock(cur_string,QR_MODE_ALPHA,last_method)){
                    generateBlock(cur_string,QR_MODE_BYTE,last_method);
                }
            }
            tmp.blocks.push_back(last_method);
            tmp.len = tmp.sum_len();
            strategy.push_back(tmp);
        }
        else {
            size_t str_len = cur_string.length();
            encodingMethods previous;
            encodingMethods new_method;
            new_method.len = error_mode_occur;

            /**divide into 2 segment*/
            for(size_t j = 0; j <  str_len; j++ ){
                /**first is previous*/
                previous = strategy[j];
                /**second is new */
                std::string sub_string = cur_string.substr(j,str_len-j);
                methodBlock blocks[3];//numeric,alpha,byte;
                generateBlock(sub_string,QR_MODE_NUM,blocks[0]);//numeric);
                generateBlock(sub_string,QR_MODE_ALPHA,blocks[1]);//alpha);
                generateBlock(sub_string,QR_MODE_BYTE,blocks[2]);//byte);
                int index = 0 ;
                int min_len = error_mode_occur;
                for(int p = 0 ; p < 3 ; p ++ ){
                    /**update the min index and min total len*/
                    if(blocks[p].payload_len+previous.len<min_len){
                        index = p;
                        min_len = blocks[p].payload_len+previous.len;
                    }
                }
                previous.blocks.push_back(blocks[index]);
                previous.len = previous.sum_len();
                if(previous.len<new_method.len){
                    new_method=previous;
                }
            }
            strategy.push_back(new_method);
        }
    }
    encodingMethods result = strategy[strategy.size()-1];

    for(size_t i = 0 ; i < result.blocks.size() ; i++){
        for (int j = 0; j < result.blocks[i].payload_len; ++j) {
            output[output_len++]=result.blocks[i].payload[j];
        }
    }
    return true;
}

/**name  @
 * param @
 * func  @ Pad the original string to fulfill all data codewords
 *          Detailed explanation can be found at @https://www.thonky.com/qr-code-tutorial/data-encoding
 */
void QREncoder::padBitStream(){
    /**total data codeword = total codeword - total ecc codeword*/
    int total_data = version_info->total_codewords - cur_ecc_params->ecc_codewords*(cur_ecc_params->num_blocks_in_G1+cur_ecc_params->num_blocks_in_G2);
    total_data *=8;
    int pad_num = total_data - payload_len;
    if(pad_num<0)
        return ;
    if(payload_len == 0)
        return;
    else if (payload_len <= 4){
        /** Add a Terminator of 0s (for padding)*/
        std::string pad = decToBin(0,payload_len);
        loadString(payload,payload_len,pad, true);
    }
    else{
        /** Add a Terminator of 0s*/
        loadString(payload,payload_len,"0000", true);
        int i = payload_len%8;
        if(i!=0){
            /**Add More 0s to Make the Length a Multiple of 8*/
            std::string pad = decToBin(0,8-i);
            loadString(payload,payload_len,pad, true);
        }
        pad_num = total_data - payload_len;
        CV_Assert(pad_num>=0);
        if( payload_len > 0 ){
            /**Add Pad Bytes if the String is Still too Short*/
            std::string pad_pattern[2] = {"11101100","00010001"};
            int num = pad_num/8;
            for(int j = 0 ; j < num ; j++){
                /**switch between two padding mode*/
                loadString(payload,payload_len,pad_pattern[j%2], true);
            }
        }
    }
    return;
}

/**name  @
 * param @
 * func  @ Convert the character string into a bit stream.
 */
void QREncoder::stringToBits(){
    switch (mode_type){
        case QR_MODE_NUM:
            encodeNumeric(input_info,payload,payload_len);
            break;
        case QR_MODE_ALPHA:
            encodeAlpha(input_info,payload,payload_len);
            break;
        case QR_MODE_STRUCTURE:
            //err = decode_structure(ptr);
            break;
        case QR_MODE_BYTE:
            encodeByte(input_info,payload,payload_len);
            break;
        case QR_MODE_KANJI:
            encodeKanji(input_info,payload,payload_len);
            break;
        case QR_MODE_ECI:
            //err = decode_eci(ptr);
            break;
        case QR_MODE_FNC1FIRST:
            //fnc1_first = true;
            break;
        case QR_MODE_FNC1SECOND:
            //fnc1_second_AI = getBits(8,ptr);
            //fnc1_second = true;
            break;
        default:
            encodeAuto(input_info,payload,payload_len);
            break;

    }
    padBitStream();
    return ;
};

/**name  @
 * param @ vector<Mat>& data_blocks,vector<Mat>& ecc_blocks
 *        ( data and ecc codeword blocks by sequence, every item is a poly of codeword block)
 * func  @ Get the data blocks by the bit stream and calculate the ecc codeword blocks.
 */
void QREncoder::eccGenerate(vector<Mat>& data_blocks,vector<Mat>& ecc_blocks){
    int EC_codewords = cur_ecc_params->ecc_codewords;
    /**read position*/
    int pay_index = 0;
    /**length between two groups*/
    int is_not_equal = cur_ecc_params->data_codewords_in_G2 - cur_ecc_params->data_codewords_in_G1 ;
    /**generator for ecc code */
    Mat G_x = polyGenerator(EC_codewords);
    /**total blocks number*/
    int blocks = cur_ecc_params->num_blocks_in_G2+cur_ecc_params->num_blocks_in_G1;
    for(int i = 0 ; i < blocks ; i++){
        /**current data and ecc block*/
        Mat Block_i,ecc_i;
        int block_len = 0;
        if(i<cur_ecc_params->num_blocks_in_G1){
            block_len = cur_ecc_params->data_codewords_in_G1;
        }
        else{
            block_len = cur_ecc_params->data_codewords_in_G2;
        }
        /**get the data codeword*/
        Block_i = Mat(Size(block_len,1),CV_8UC1,Scalar(0));
        for(int j = 0 ;j < block_len; j++){
            Block_i.ptr(0)[block_len-1-j] = (uchar)getBits(8,payload,pay_index);
        }
        /**get the ecc block by division*/
        Mat dividend ;
        Mat shift = Mat(Size(EC_codewords,1),CV_8UC1,Scalar(0));
        hconcat(shift,Block_i,dividend);
        ecc_i = gfPolyDiv(dividend,G_x,EC_codewords);

        /**align the data codeword by padding last 0 if G2 is longer than G1*/
        if(is_not_equal&&i<cur_ecc_params->num_blocks_in_G1){
            Mat padding = Mat(Size(1,1),CV_8UC1,Scalar(0));
            hconcat(padding,Block_i,Block_i);
        }

        data_blocks.push_back(Block_i);
        ecc_blocks.push_back(ecc_i);

    }
    return ;
}

/**name  @
 * param @ const vector<Mat>& data_blocks,const vector<Mat>& ecc_blocks
 *        ( data and ecc codeword blocks by sequence, every item is a poly of codeword block)
 * func  @ Rearrange the all the codewords in vertical sequence for the encoding.
 */
void QREncoder::rearrangeBlocks(const vector<Mat>& data_blocks,const vector<Mat>& ecc_blocks){
    int blocks = cur_ecc_params->num_blocks_in_G2+cur_ecc_params->num_blocks_in_G1;
    int col_border = max(cur_ecc_params->data_codewords_in_G2,cur_ecc_params->data_codewords_in_G1);

    /**total ecc codeword num*/
    int total_num = blocks*(col_border+cur_ecc_params->ecc_codewords);
    int is_not_equal = cur_ecc_params->data_codewords_in_G2 - cur_ecc_params->data_codewords_in_G1 ;
    int rearranged_len = 0 ;
    int data_col = data_blocks[0].cols-1;
    int ecc_col = ecc_blocks[0].cols-1;
    /**rearrange process*/
    for(int i = 0 ; i < total_num; i++ ){
        int cur_col = i / blocks ;
        int cur_row = i % blocks ;

        std::string bits;
        uint8_t tmp = 0;
        /**for data codeword */
        if(cur_col < col_border){
            /**read from data codeword*/
            if(is_not_equal && cur_col==cur_ecc_params->data_codewords_in_G2-1 && cur_row < cur_ecc_params->num_blocks_in_G1){
                /**G2 is longer than G1 , we need to ignore codeword padded before*/
                continue;
            }
            else{
                /**load in the final data array*/
                bits = decToBin(data_blocks[cur_row].ptr(0)[data_col-cur_col],8);
                tmp = data_blocks[cur_row].ptr(0)[data_col-cur_col];
            }
        }
        else{
            /**for ecc codeword */
            int index = ecc_col-(cur_col-col_border);
            /**read from ecc codeword*/
            bits = decToBin(ecc_blocks[cur_row].ptr(0)[index],8);
            tmp = ecc_blocks[cur_row].ptr(0)[index];
        }
        rearranged_data[rearranged_len++]=tmp;
    }
    return ;
}

/**name  @
 * param @
 * func  @ Mask the QRcode by corresponding mask type
 */
void QREncoder::maskData(){
    for(int i= 0;i<version_size;i++){
        for(int j= 0;j<version_size;j++){
            /**ignore the Reserved Area*/
            if(original.ptr(i)[j]==invalid_region_value)
                continue;
                /**unmask*/
            else if((mask_type==0&&!((i + j) % 2)) ||
                    (mask_type==1&&!(i % 2)) ||
                    (mask_type==2&&!(j % 3)) ||
                    (mask_type==3 && !((i + j) % 3 )) ||
                    (mask_type==4&&!(((i / 2) + (j / 3)) % 2)) ||
                    (mask_type==5&&!((i * j) % 2 + (i * j) % 3))||
                    (mask_type==6&&!(((i * j) % 2 + (i * j) % 3) % 2))||
                    ((mask_type==7 && !(((i * j) % 3 + (i + j) % 2) % 2)))
                    ){
                masked_data.ptr(i)[j] = original.ptr(i)[j]^255;
            }
            else
                /**the same*/
                masked_data.ptr(i)[j] = original.ptr(i)[j];
        }
    }
    return;
}

/**name  @
 * param @
 * func  @ write loactor pattern , timing pattern and alignment position
 */
void QREncoder::writeReservedArea(){
    /*get mask pattern according to the format*/
    int locator_position[2] = { 3 , version_size -1 -3 };
    /**draw locator pattern*/
    for (int a = 0 ; a < 2 ; a++ ){
        for(int p = 0 ; p < 2 ; p++){
            if(a==1 && p==1)
                continue;
            int x=locator_position[a];
            int y=locator_position[p];

            for(int i=-5;i<=5;i++)
                for(int j=-5;j<=5;j++){
                    /**can beyond the boundary*/
                    if(x+i<0 || x+i >=version_size || y+j < 0 || y+j >=version_size)
                        continue;
                    if((( j == 2 || j == -2)  &&  -2<= i && i <=2 )||
                       ( -2 <= j && j <=  2   && ( i == 2 || i == -2))|| /**loactor pattern*/
                       abs(i)==4 || abs(j)==4 )/**quiet zone*/
                        masked_data.ptr(x+i)[y+j]=255;
                    else
                        masked_data.ptr(x+i)[y+j]=0;

                    if(version_level<7){
                        /**format area*/
                        if ((y == locator_position[1] && j==-5) || (x == locator_position[1] && i==-5)){
                            continue;
                        }else{
                            original.ptr(x+i)[y+j]=invalid_region_value;
                        }
                    }
                    else{
                        original.ptr(x+i)[y+j]=invalid_region_value;
                    }
                }
        }
    }
    /** Dark point*/
    int x=locator_position[1]-4;
    int y=locator_position[0]+5;
    masked_data.ptr(x)[y]=0;
    original.ptr(x)[y]=invalid_region_value;

    for(int i= 0;i<version_size;i++){
        for(int j= 0;j<version_size;j++){
            /** i for row and j for col*/
            if(original.ptr(i)[j]==invalid_region_value)
                continue;
            if ((i == 6 || j == 6)){
                /** Exclude timing patterns */
                original.ptr(i)[j]=invalid_region_value;
                if(((i == 6) && (j-7)%2 == 0)||/** up horizontal line*/
                   ((j == 6) && ((i-7)%2 == 0)))/** left vertical line*/
                    masked_data.ptr(i)[j] = 255;
                else
                    masked_data.ptr(i)[j] = 0;
            }
            /**version_level information*/
            if (version_level >= 7) {
                if ((i < 6 && j + 11 >= version_size)||(i + 11 >= version_size && j < 6)){
                    original.ptr(i)[j]=invalid_region_value;
                    /**/
                }
            }

        }
    }
    /** Exclude alignment patterns */
    for (int a = 0; a < max_alignment && version_info->alignment_pattern[a]; a++) {
        for (int p = a; p < max_alignment && version_info->alignment_pattern[p]; p++) {
            x=version_info->alignment_pattern[a];
            y=version_info->alignment_pattern[p];
            /*the alignment patterns MUST NOT overlap the finder patterns or separators*/
            if(original.ptr(x)[y]==invalid_region_value)
                continue;
            for(int i=-2;i<=2;i++)
                for(int j=-2;j<=2;j++){
                    original.ptr(x+i)[y+j]=invalid_region_value;
                    if(j==0 && i==0)
                        masked_data.ptr(x+i)[y+j]=0;
                    else if( j==-2 || j==2 || i ==-2 || i ==2 ){
                        masked_data.ptr(x+i)[y+j]= 0;
                    }
                    else
                        masked_data.ptr(x+i)[y+j]=255;
                }
        }
    }
    return ;
}

/**name  @
 * param @ int x, int y ( current pixel postion), int& count(the number of current bit)
 * func  @ write bit into the QR code
 */
void QREncoder::writeBit(int x, int y, int& count){
    /**the bitpos^th bit of the  bytepos^th codeword*/
    int bytepos = count >> 3;/*equal to count/8 */
    int bitpos  = count & 7 ;/*equal to count%8 */
    /**judge the reserved area*/
    if (original.ptr(y)[x]==invalid_region_value){
        return ;
    }
    int v = ((rearranged_data[bytepos] & (0x80 >> bitpos)) == 0);
    /** first read,first lead*/
    if (v){
        original.ptr(y)[x] = 255;
    }
    else{
        original.ptr(y)[x] = 0;
    }
    count++;
}
/**name  @
 * param @
 * func  @ write data into the QRcode by zig-zag method
 */
void QREncoder::writeData(){
    int y = version_size - 1;
    int x = version_size - 1;
    int dir = -1;
    int count = 0;
    while (x > 0) {
        if (x == 6)
            x--;
        /**write*/
        writeBit( x,  y, count);
        writeBit( x-1,  y, count);

        y += dir;
        /**change direction when meets border*/
        if (y < 0 || y >= version_size ) {
            dir = -dir;
            x -= 2;
            y += dir;
        }
    }
}
/**name  @
 * param @
 * func  @ write the format and verion information in the Reserved Area
 */
void QREncoder::fillReserved(){
    /**write to the left-bottom and upper-right */
    int i  ;
    /**format area*/
    if(version_level<7){
        /*left-bottom 0-7*/
        for (i = 0; i < 7; i++){
            /**read from pst (code->size - 1 - i,8)*/
            masked_data.ptr(version_size - 1 - i)[8]= 255*(int)(format.ptr(0)[max_format_length-1 -i] == 0) ;
        }
        /**upper-right 7-14*/
        for (i = 0; i < 8; i++){
            masked_data.ptr(8)[version_size - 8 + i]=255*(int)(format.ptr(0)[max_format_length-1 -(7+i)] == 0);
        }
        /**write the second format at the upper-left*/
        static const int xs[max_format_length] = {
                8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0
        };
        static const int ys[max_format_length] = {
                0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8
        };
        for (i = max_format_length-1; i >= 0; i--) {
            masked_data.ptr<uint8_t>(ys[i])[xs[i]] = 255*(int)( format.ptr(0)[i] == 0 );
        }
    }
    return;
}
/**name  @
 * param @
 * func  @ Write the bits into QR image
 */
void QREncoder::structureFinalMessage(){
    writeReservedArea();
    writeData();
    maskData();
    fillReserved();
    return ;
}

Mat QREncoder::QRcodeGenerate(){
    vector<Mat> data_blocks,ecc_blocks;
    Mat tmp ;

    formatGenerate();
    stringToBits();
    eccGenerate(data_blocks,ecc_blocks);
    rearrangeBlocks(data_blocks,ecc_blocks);
    structureFinalMessage();

    resize(masked_data,tmp,Size(600,600),0,0,INTER_AREA);
    int border = int(tmp.cols*0.05);
    copyMakeBorder(tmp, tmp, border, border, border, border, BORDER_CONSTANT, Scalar(255));
    return tmp;
}

}  // namespace