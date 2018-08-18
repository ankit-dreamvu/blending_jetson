// g++ blend_all.cpp -o rblend `pkg-config --cflags --libs opencv` -std=c++11 -O3
// g++ test_main.cpp -o dblend `pkg-config --cflags --libs opencv` -std=c++11 -g

# if 1
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/types.hpp"
#include "test_precomp.hpp"
# else
#include <string>
#include <stdio.h>
#include <iostream>
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/stitching/detail/blenders.hpp"
#include <sys/time.h>
#include <cstdlib>
#include <dirent.h> 
#include <time.h>
#include <sstream>
#include <fstream>
#include <ctime>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

// #include "opencv2/gpu/gpu.hpp"
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/cudaimgproc.hpp>

# include <memory>
# include <stdio.h>
# include <stdlib.h>

# endif
using namespace cv;
using namespace cv::detail;
using namespace std;
using namespace cv::cuda;

bool try_use_gpu = false;
Stitcher::Mode mode = Stitcher::PANORAMA;

int img_unique_val = 0;
string result_name = "result.jpg";

int split_val = 1;          //spilt division value
double split_ratio = 0.125;


unsigned char* concat_panorama = 0;


int DESIRED_BLEND_WIDTH = 100;
bool TRY_GPU = true;

struct TimeLogger
{
    struct EventPoint
    {
        int line_number;
        std::string event_name;
        timeval time;
    };
    timeval start;
    std::vector<EventPoint> time_points;

    void Start()
    {
        gettimeofday(&start, NULL);
        time_points.clear();
    }

    void Log(int line, std::string name)
    {
        EventPoint ep;
        ep.line_number = line;
        ep.event_name = name;
        gettimeofday(&ep.time, NULL);
        time_points.push_back(ep);
    }

    double Diff_ms(timeval start1, timeval end)
    {
        double seconds = end.tv_sec - start1.tv_sec;
        double useconds = end.tv_usec - start1.tv_usec;

        double mtime = ((seconds)* 1000.0 + useconds / 1000.0) + 0.5;

        return mtime;
    }

    double Display()
    {
        //std::cout << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
        timeval last = start;
        for (size_t i = 0; i < time_points.size(); i++)
        {
            double ms = Diff_ms(last, time_points[i].time);
            double ams = Diff_ms(start, time_points[i].time);
            std::cout << ams << " is " << " Accumulated time......." <<
                ms << " ms for " <<  time_points[i].event_name << "\n";
            last = time_points[i].time;
        }

		return Diff_ms(start, last);

    }
};

TimeLogger g_time_logger;

# define LOG(x) g_time_logger.Log(__LINE__, #x)

bool g_bUseGPU = true;
MultiBandBlender *g_pBlender = 0;// blender1(true, 5);

void blend_test(const char* in1, const char* in2, const char* out)
{
    Mat image1 = imread(in1);
    Mat image2 = imread(in2);

    Mat conv;
    image1.convertTo(conv, CV_16S);
    
    Mat concatenated_image1, concatenated_image2;
    Mat image1s, image2s;
    Mat img1 = image1(Rect(2 * (image1.cols) / 3, 0, image1.cols / 3, image1.rows));
    Mat img2 = image2(Rect(0, 0, image2.cols / 3, image2.rows));

    img1.convertTo(image1s, CV_16S);
    img2.convertTo(image2s, CV_16S);

    Mat mask1(image1s.size(), CV_8U);
    mask1(Rect(0, 0, mask1.cols / 2, mask1.rows)).setTo(255);
    mask1(Rect(mask1.cols / 2, 0, mask1.cols - mask1.cols / 2, mask1.rows)).setTo(0);

    Mat mask2(image2s.size(), CV_8U);
    mask2(Rect(0, 0, mask2.cols / 2, mask2.rows)).setTo(0);
    mask2(Rect(mask2.cols / 2, 0, mask2.cols - mask2.cols / 2, mask2.rows)).setTo(255);

    Mat result_s1, result_mask1;

    g_pBlender->prepare(Rect(0, 0, max(image1s.cols, image2s.cols), max(image1s.rows, image2s.rows)));
    g_pBlender->feed(image1s, mask1, Point(0, 0));
    g_pBlender->feed(image2s, mask2, Point(0, 0));
    g_pBlender->blend(result_s1, result_mask1);


    imwrite(out, result_s1);
}



void DumpTga(const char* fileName, int width, int height, int bpp, unsigned char* data)
{
    unsigned char header[18] = { 0 };
    for (int i = 0; i < 18; i++) header[i] = 0;

    header[2] = 2;

    header[12] = width & 255;
    header[13] = width >> 8;

    header[14] = height & 255;
    header[15] = height >> 8;

    header[16] = bpp * 8;

    FILE* fp = fopen(fileName, "wb");
    fwrite(header, 18, 1, fp);
    fwrite(data, width*height, bpp, fp);
    fclose(fp);
}


void DumpTga(const char* fileName, Mat m)
{
    DumpTga(fileName, m.cols, m.rows, 3, m.data);
}

void Copy2D(unsigned char* dst, int dStride, unsigned char* src, int sStride, int width, int rows, bool flip = false)
{

# if 0
    if (flip)
    {
        for (int i = 0; i < rows; i++)
        {
            memcpy(dst + (rows - i - 1)*dStride, src + i*sStride, width);
        }
        return;
    }
    for (int i = 0; i < rows; i++)
    {
        memcpy(dst + i*dStride, src + i*sStride, width);
    }
# else
	(void)flip;
	cudaMemcpy2D(dst, dStride, src, sStride, width, rows, cudaMemcpyDefault);
# endif
}


class DV_Blender : private MultiBandBlender
{
    DV_Blender();

public:

    DV_Blender(bool use_gpu, int levels, int weight_type = 5) : MultiBandBlender(use_gpu, levels, weight_type) {}

    void DV_Feed(InputArray _img, InputArray mask, Point tl);

    void DV_Prepare(Rect dst_roi);

    void DV_Blend(InputOutputArray dst, InputOutputArray dst_mask);
};


void DV_Blender::DV_Feed(InputArray _img, InputArray mask, Point tl)
{
    MultiBandBlender::feed(_img, mask, tl);
}


void DV_Blender::DV_Prepare(Rect dst_roi)
{
    MultiBandBlender::prepare(dst_roi);
}


void DV_Blender::DV_Blend(InputOutputArray dst, InputOutputArray dst_mask)
{
    MultiBandBlender::blend(dst, dst_mask);
}



int ConcatFrames(unsigned char* in1, unsigned char* b12, unsigned char* in2, unsigned char* b23,
                 unsigned char* in3, unsigned char* b34, unsigned char* in4, unsigned char* dst,
                 int cols, int rows)
{
    int border1 = cols / 3;
    int border2 = 2 * cols / 3;
    int stride = cols * 3;
    in1 += 0;
    in2 += border1 * 3;
    in3 += border1 * 3;
    in4 += border1 * 3;

    //int blendSize = cols / 3;
    int originalSize = border2 - border1 + 1;

    int size1 = border2 * 3;
    int size2 = border1 * 3;
    int size3 = originalSize * 3;
    //unsigned char* back = dst;
    for (int i = 0; i < rows; i++)
    {
        memcpy(dst, in1, size1); in1 += stride; dst += size1;
        memcpy(dst, b12, size2); b12 += size2;  dst += size2;
        memcpy(dst, in2, size3); in2 += stride; dst += size3;
        memcpy(dst, b23, size2); b23 += size2;  dst += size2;
        memcpy(dst, in3, size3); in3 += stride; dst += size3;
        memcpy(dst, b34, size2); b34 += size2;  dst += size2;
        memcpy(dst, in4, size3); in4 += stride; dst += size3;
    }

    return border2 + 3 * (originalSize + border1);

}

Mat multiband_blending(Mat image1, Mat image2)
{

    //cout<<"started the blend\n";
    Mat concatenated_image1, concatenated_image2;
    Mat image1s, image2s;
    Mat img1 = image1;//(Rect(1024,0,256,image1.rows));
    Mat img2 = image2;//(Rect(0,0,256,image2.rows));
    /* Mat img1 = image1(Rect(4*(image1.cols)/5,0,image1.cols/5,image1.rows));
    Mat img2 = image2(Rect(0,0,image2.cols/5,image2.rows));*/
    //cout<<"defined rois\n";

    img1.convertTo(image1s, CV_16S);
    img2.convertTo(image2s, CV_16S);

    // cv::Size img_size = cv::Size(485,1300);
    static Mat mask1(img1.size(), CV_8U);
    static Mat mask2(img2.size(), CV_8U);

    static int firstTime = 1;
    if (firstTime)
    {

# if 0
		for (int i = 0; i < mask1.cols; i++)
		{
			int w = i * 255 / mask1.cols;
			mask1(Rect(mask1.cols - i - 1, 0, 1, mask1.rows)).setTo(w);
			mask2(Rect(i, 0, 1, mask2.rows)).setTo(w);

		}
        //cout<<"defined mask\n";
# else
		mask1(Rect(0, 0, mask1.cols / 2, mask1.rows)).setTo(255);
		mask1(Rect(mask1.cols / 2, 0, mask1.cols - mask1.cols / 2, mask1.rows)).setTo(0);

		mask2(Rect(0, 0, mask2.cols / 2, mask2.rows)).setTo(0);
		mask2(Rect(mask2.cols / 2, 0, mask2.cols - mask2.cols / 2, mask2.rows)).setTo(255);
		//cout<<"defined mask\n";

# endif
        firstTime = 0;

		g_pBlender->prepare(Rect(0, 0, max(image1s.cols, image2s.cols), max(image1s.rows, image2s.rows)));
		LOG(prepare);
    }
	if (!g_bUseGPU)
	{
		g_pBlender->prepare(Rect(0, 0, max(image1s.cols, image2s.cols), max(image1s.rows, image2s.rows)));
		LOG(prepare);
	}

    //MultiBandBlender blender1(true, 5);
    //DV_Blender blender1(TRY_GPU, 5);
    //cout<<"initialized blender\n";

    g_pBlender->feed(image1s, mask1, Point(0, 0));
	LOG(feed1);

    g_pBlender->feed(image2s, mask2, Point(0, 0));
	LOG(feed2);

    Mat result_s1, result_mask1;
    g_pBlender->blend(result_s1, result_mask1);
	LOG(blend);

    Mat result1;
    result_s1.convertTo(result1, CV_8U);
	LOG(convert);

    return result1;
}



int test_main()
{
    blend_test("/krkjp/in/image_1.png",
               "/krkjp/in/image_2.png",
               "/krkjp/in/reused_blend_12.png");

    blend_test("/krkjp/in/image_2.png",
               "/krkjp/in/image_3.png",
               "/krkjp/in/reused_blend_23.png");

    return 0;
}

int main(int argc, char* argv[])
{
    int desired_blend_width = 200;
    if (argc > 1)
    {
        desired_blend_width = atoi(argv[1]);
        if (desired_blend_width < 0 || desired_blend_width > 250)
            desired_blend_width = 250;
    }
	if (argc > 2)
	{
		g_bUseGPU = (atoi(argv[2]) != 0);
	}
	MultiBandBlender blender(g_bUseGPU, 5);
	g_pBlender = &blender;
    int frames = 30;

    Mat image1, image2, image3, image4;

    concat_panorama = new unsigned char[4096 * 2048 * 4];
	double processTime = 0.0;
    for (int i = 0; i < frames; i++)
    {
        Mat frame;

        const char* path = "/home/nvidia/Documents/ALIA-ImageProcessing/dataset/video_extracts/";
        char fileName[1024];
        sprintf(fileName, "%s/image_1/%06d.jpg", path, i);
        image1 = imread(fileName);
        sprintf(fileName, "%s/image_2/%06d.jpg", path, i);
        image2 = imread(fileName);
        sprintf(fileName, "%s/image_3/%06d.jpg", path, i);
        image3 = imread(fileName);
        sprintf(fileName, "%s/image_4/%06d.jpg", path, i);
        image4 = imread(fileName);

        image1.convertTo(image1, CV_8UC3);
        image2.convertTo(image2, CV_8UC3);
        image3.convertTo(image3, CV_8UC3);
        image4.convertTo(image4, CV_8UC3);

        //DumpTga("1.tga", image1);

        g_time_logger.Start();

        //some border width is ommitted on the both sides,
        //so that input image width = (blendWidth + 1024 + blendWidth)
        int single_image_width = 1024;
        int ommitted_border_width = (image1.cols - single_image_width) / 2;
        int rows =  image1.rows;
        
        //   discard    minor-blend   major-blend      use-as-it-is     major-blend  minor-blend  discard
        // x0........x1...........x2.............x3..................x4............x5...........x6.......x7
        //int x0 = 0;
        int x1 = ommitted_border_width - desired_blend_width/2;
        int x2 = ommitted_border_width;
        int x3 = x1 + desired_blend_width;
        int x4 = x2 + single_image_width - desired_blend_width / 2;
        //int x5 = x2 + single_image_width;
        //int x6 = x4 + desired_blend_width;
        //int x7 = image1.cols;

        int in_width = x4 - x3;
        int blend_width = desired_blend_width;//256;

        Rect leftPortion = Rect(x1, 0, blend_width, rows);
        Rect rightPortion = Rect(x4, 0, blend_width, rows);

        Mat b12, b23, b34, b41;

//#pragma omp parallel
        {
//#pragma omp task
            b12 = multiband_blending(image1(rightPortion), image2(leftPortion));

//#pragma omp task
            b23 = multiband_blending(image2(rightPortion), image3(leftPortion));

//#pragma omp task
            b34 = multiband_blending(image3(rightPortion), image4(leftPortion));

//#pragma omp task
            b41 = multiband_blending(image4(rightPortion), image1(leftPortion));
        }

        unsigned char* in_1 = image1.data;
        unsigned char* in_2 = image2.data;
        unsigned char* in_3 = image3.data;
        unsigned char* in_4 = image4.data;

        unsigned char* b1_2 = b12.data;
        unsigned char* b2_3 = b23.data;
        unsigned char* b3_4 = b34.data;
        unsigned char* b4_1 = b41.data;

        unsigned char* dst = concat_panorama;//.data;
        int dstOffset = 0;
        int dstStride = (b12.cols + in_width) * 4 * 3;
        int srcStride = image1.cols * 3;
        int blendStride = b12.cols * 3;

        LOG(BLEND-DONE);


        //   discard    minor-blend   major-blend      use-as-it-is     major-blend  minor-blend  discard
        // x0........x1...........x2.............x3..................x4............x5...........x6.......x7
        Copy2D(dst + dstOffset, dstStride, in_1 + x3 * 3, srcStride, in_width * 3, b12.rows);  dstOffset += in_width * 3;
        Copy2D(dst + dstOffset, dstStride, b1_2, blendStride, blendStride, b12.rows);  dstOffset += blendStride;
        Copy2D(dst + dstOffset, dstStride, in_2 + x3 * 3, srcStride, in_width * 3, b12.rows);  dstOffset += in_width * 3;
        Copy2D(dst + dstOffset, dstStride, b2_3, blendStride, blendStride, b12.rows);  dstOffset += blendStride;
        Copy2D(dst + dstOffset, dstStride, in_3 + x3 * 3, srcStride, in_width * 3, b12.rows);  dstOffset += in_width * 3;
        Copy2D(dst + dstOffset, dstStride, b3_4, blendStride, blendStride, b12.rows);  dstOffset += blendStride;
        Copy2D(dst + dstOffset, dstStride, in_4 + x3 * 3, srcStride, in_width * 3, b12.rows);  dstOffset += in_width * 3;
        Copy2D(dst + dstOffset, dstStride, b4_1, blendStride, blendStride, b12.rows);  dstOffset += blendStride;

        LOG(concat);
		processTime += g_time_logger.Display();
		if (i == 0) processTime = 0.0;

# if  1  //def DUMP_PANORAMA
        Mat panorama(b12.rows, dstStride / 3, CV_8UC3, concat_panorama);
        sprintf(fileName, "out/%06d.png", i);
        imwrite(fileName, panorama);
        //sprintf(fileName, "out/%06d.tga", i);
        //DumpTga(fileName, panorama);
        //return 0;
# endif


        cout << "blend_width = " << desired_blend_width << (g_bUseGPU ? " GPU" : " CPU") << "------------------------------------------------------------\n";
    }

    delete[] concat_panorama;
	double avg = processTime / (double)(frames-1);
	double fps = 1000.0 / avg;
	printf("\n\n\nAVERAGE TIME = %lf\n\n FPS = %lf\n\n", avg, fps);
    return 0;
}

