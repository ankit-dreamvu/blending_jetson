//  nvcc ./memcpy.cu -arch=sm_62 && ./a.out

# include <stdio.h>
# include <sys/time.h>
# include <cstdlib>
# include <iostream>
# include <ctime>
# include <vector>
# include <sys/types.h>
# include <cuda_runtime.h>

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
				ms << " ms for " << time_points[i].event_name << "\n";
			last = time_points[i].time;
		}

		return Diff_ms(start, last);

	}
};

TimeLogger g_time_logger;

# define LOG(x) g_time_logger.Log(__LINE__, #x)


FILE* Open(const char* fileName, const char* mode)
{
# ifdef _WIN32
	FILE* fp = 0;
	fopen_s(&fp, fileName, mode);
	return fp;
# else
	return fopen(fileName, mode);
# endif
}


void DumpTga(const char* fileName, int width, int height, int bpp, unsigned char* data)
{
	unsigned char header[18] = { 0 };
	for (int i = 0; i < 18; i++) header[i] = 0;

	header[2] = (bpp == 1) ? 3 : 2;

	header[12] = width & 255;
	header[13] = width >> 8;

	header[14] = height & 255;
	header[15] = height >> 8;

	header[16] = bpp * 8;

	FILE* fp = Open(fileName, "wb");
	fwrite(header, 18, 1, fp);
	fwrite(data, width*height, bpp, fp);
	fclose(fp);
}


unsigned char* ReadTGA(const char* fileName, int& width, int &height, int &bpp, unsigned char* pixels = 0)
{
	width = height = bpp = 0;

	FILE* fp = Open(fileName, "rb");
	if (!fp) return pixels;

	unsigned char header[18] = { 0 };
	fread(header, 18, 1, fp);

	if (header[2] != 2 && header[2] != 3)
		return pixels;

	width = header[12] + header[13] * 256;
	height = header[14] + header[15] * 256;
	bpp = header[16] >> 3;



	//unsigned char* pixels = 0;
	if(!pixels) 
		pixels = new unsigned char[width*height*bpp];

	fread(pixels, 76, 1, fp);
	if (pixels[0] == 'R'
		&&	pixels[1] == 'e'
		&&	pixels[2] == 'n'
		&&	pixels[3] == 'd'
		&&	pixels[4] == 'e'
		&&	pixels[5] == 'r'
		)
	{
		fread(pixels, width*height, bpp, fp);
	}
	else
	{
		fread(pixels + 76, width*height*bpp - 76, 1, fp);

	}
	fclose(fp);

	return pixels;
}



const int width = (1 << 12);
const int height = width / 2;
const int bpp = 3;

__device__ __managed__ unsigned char img[width*height*bpp];
__device__ __managed__ unsigned char outH[width*height*bpp];
__device__ __managed__ unsigned char outV[width*height*bpp];
__device__ __managed__ unsigned char out[width*height*bpp];

__global__ void Gradient(unsigned char* img, int x, int y)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = (col + row*width)*bpp;
	col = (col + x) % width;
	row = (row + y) % height;
	unsigned char r = col & 255;
	unsigned char g = row & 255;
	unsigned char b = ((row >> 8) << 4) | (col >> 8);
	img[index] = b;
	img[index+1] = g;
	img[index+2] = r;
}

__global__ void DownScale(unsigned char* src, unsigned char* dst, int rows, int cols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int dst_index = (col + row * cols)*bpp;
	row = row << 1;
	col = col << 1;
	int src_index = (col + row * cols*2)*bpp;
	unsigned char b = src[src_index];
	unsigned char g = src[src_index + 1];
	unsigned char r = src[src_index + 2];
	dst[dst_index] = (256+b)&255;
	dst[dst_index + 1] = g;
	dst[dst_index + 2] = r;
}



__global__ void Blur(unsigned char* src, unsigned char* dst, int rows, int cols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int dst_index = (col + row * cols)*bpp;
	row = row << 1;
	col = col << 1;

	int col0 = max(col - 2, 0);
	int col1 = max(col - 1, 0);
	int col2 = col;
	int col3 = min(col + 1, cols - 1);
	int col4 = min(col + 2, cols - 1);

	int row0 = max(row - 2, 0);
	int row1 = max(row - 1, 0);
	int row2 = row;
	int row3 = min(row + 1, rows - 1);
	int row4 = min(row + 2, rows - 1);

# define C00 1
# define C01 4
# define C02 6
# define C03 4
# define C04 1

# define C10 4
# define C11 16
# define C12 24
# define C13 16
# define C14 1

# define C20 6
# define C21 24
# define C22 36
# define C23 24
# define C24 6 

# define C30 4
# define C31 16
# define C32 24
# define C33 16
# define C34 1

# define C40 1
# define C41 4
# define C42 6
# define C43 4
# define C44 1
	/*
# define R(x,y) src[bpp*(row##y*cols + col##x )]
# define G(x,y) src[bpp*(row##y*cols + col##x ) + 1]
# define B(x,y) src[bpp*(row##y*cols + col##x ) + 2]

# define DEF(c, y, x) unsigned char c##y##x = c(x,y)
# define PIX(y, x) DEF(R,y,x); DEF(G,y,x); DEF(B, y,x);

	PIX(0, 0);	PIX(0, 1);	PIX(0, 2);	PIX(0, 3);	PIX(0, 4);
	PIX(1, 0);	PIX(1, 1);	PIX(1, 2);	PIX(1, 3);	PIX(1, 4);
	PIX(2, 0);	PIX(2, 1);	PIX(2, 2);	PIX(2, 3);	PIX(2, 4);
	PIX(3, 0);	PIX(3, 1);	PIX(3, 2);	PIX(3, 3);	PIX(3, 4);
	PIX(4, 0);	PIX(4, 1);	PIX(4, 2);	PIX(4, 3);	PIX(4, 4);*/

	unsigned char R00 = src[bpp*(row0*cols*2 + col0)];
	unsigned char R01 = src[bpp*(row0*cols*2 + col1)];
	unsigned char R02 = src[bpp*(row0*cols*2 + col2)];
	unsigned char R03 = src[bpp*(row0*cols*2 + col3)];
	unsigned char R04 = src[bpp*(row0*cols*2 + col4)];

	unsigned char R10 = src[bpp*(row1*cols*2 + col0)];
	unsigned char R11 = src[bpp*(row1*cols*2 + col1)];
	unsigned char R12 = src[bpp*(row1*cols*2 + col2)];
	unsigned char R13 = src[bpp*(row1*cols*2 + col3)];
	unsigned char R14 = src[bpp*(row1*cols*2 + col4)];

	unsigned char R20 = src[bpp*(row2*cols*2 + col0)];
	unsigned char R21 = src[bpp*(row2*cols*2 + col1)];
	unsigned char R22 = src[bpp*(row2*cols*2 + col2)];
	unsigned char R23 = src[bpp*(row2*cols*2 + col3)];
	unsigned char R24 = src[bpp*(row2*cols*2 + col4)];

	unsigned char R30 = src[bpp*(row3*cols*2 + col0)];
	unsigned char R31 = src[bpp*(row3*cols*2 + col1)];
	unsigned char R32 = src[bpp*(row3*cols*2 + col2)];
	unsigned char R33 = src[bpp*(row3*cols*2 + col3)];
	unsigned char R34 = src[bpp*(row3*cols*2 + col4)];

	unsigned char R40 = src[bpp*(row4*cols*2 + col0)];
	unsigned char R41 = src[bpp*(row4*cols*2 + col1)];
	unsigned char R42 = src[bpp*(row4*cols*2 + col2)];
	unsigned char R43 = src[bpp*(row4*cols*2 + col3)];
	unsigned char R44 = src[bpp*(row4*cols*2 + col4)];


	unsigned char G00 = src[1+bpp*(row0*cols*2 + col0)];
	unsigned char G01 = src[1+bpp*(row0*cols*2 + col1)];
	unsigned char G02 = src[1+bpp*(row0*cols*2 + col2)];
	unsigned char G03 = src[1+bpp*(row0*cols*2 + col3)];
	unsigned char G04 = src[1+bpp*(row0*cols*2 + col4)];

	unsigned char G10 = src[1+bpp*(row1*cols*2 + col0)];
	unsigned char G11 = src[1+bpp*(row1*cols*2 + col1)];
	unsigned char G12 = src[1+bpp*(row1*cols*2 + col2)];
	unsigned char G13 = src[1+bpp*(row1*cols*2 + col3)];
	unsigned char G14 = src[1+bpp*(row1*cols*2 + col4)];

	unsigned char G20 = src[1+bpp*(row2*cols*2 + col0)];
	unsigned char G21 = src[1+bpp*(row2*cols*2 + col1)];
	unsigned char G22 = src[1+bpp*(row2*cols*2 + col2)];
	unsigned char G23 = src[1+bpp*(row2*cols*2 + col3)];
	unsigned char G24 = src[1+bpp*(row2*cols*2 + col4)];

	unsigned char G30 = src[1+bpp*(row3*cols*2 + col0)];
	unsigned char G31 = src[1+bpp*(row3*cols*2 + col1)];
	unsigned char G32 = src[1+bpp*(row3*cols*2 + col2)];
	unsigned char G33 = src[1+bpp*(row3*cols*2 + col3)];
	unsigned char G34 = src[1+bpp*(row3*cols*2 + col4)];

	unsigned char G40 = src[1+bpp*(row4*cols*2 + col0)];
	unsigned char G41 = src[1+bpp*(row4*cols*2 + col1)];
	unsigned char G42 = src[1+bpp*(row4*cols*2 + col2)];
	unsigned char G43 = src[1+bpp*(row4*cols*2 + col3)];
	unsigned char G44 = src[1+bpp*(row4*cols*2 + col4)];


	unsigned char B00 = src[2+bpp*(row0*cols*2 + col0)];
	unsigned char B01 = src[2+bpp*(row0*cols*2 + col1)];
	unsigned char B02 = src[2+bpp*(row0*cols*2 + col2)];
	unsigned char B03 = src[2+bpp*(row0*cols*2 + col3)];
	unsigned char B04 = src[2+bpp*(row0*cols*2 + col4)];

	unsigned char B10 = src[2+bpp*(row1*cols*2 + col0)];
	unsigned char B11 = src[2+bpp*(row1*cols*2 + col1)];
	unsigned char B12 = src[2+bpp*(row1*cols*2 + col2)];
	unsigned char B13 = src[2+bpp*(row1*cols*2 + col3)];
	unsigned char B14 = src[2+bpp*(row1*cols*2 + col4)];

	unsigned char B20 = src[2+bpp*(row2*cols*2 + col0)];
	unsigned char B21 = src[2+bpp*(row2*cols*2 + col1)];
	unsigned char B22 = src[2+bpp*(row2*cols*2 + col2)];
	unsigned char B23 = src[2+bpp*(row2*cols*2 + col3)];
	unsigned char B24 = src[2+bpp*(row2*cols*2 + col4)];

	unsigned char B30 = src[2+bpp*(row3*cols*2 + col0)];
	unsigned char B31 = src[2+bpp*(row3*cols*2 + col1)];
	unsigned char B32 = src[2+bpp*(row3*cols*2 + col2)];
	unsigned char B33 = src[2+bpp*(row3*cols*2 + col3)];
	unsigned char B34 = src[2+bpp*(row3*cols*2 + col4)];

	unsigned char B40 = src[2+bpp*(row4*cols*2 + col0)];
	unsigned char B41 = src[2+bpp*(row4*cols*2 + col1)];
	unsigned char B42 = src[2+bpp*(row4*cols*2 + col2)];
	unsigned char B43 = src[2+bpp*(row4*cols*2 + col3)];
	unsigned char B44 = src[2+bpp*(row4*cols*2 + col4)];

# define RC(yx) R##yx*C##yx
# define GC(yx) G##yx*C##yx
# define BC(yx) B##yx*C##yx


	short R0 = RC(00) + RC(01) + RC(02) + RC(03) + RC(04);
	short R1 = RC(10) + RC(11) + RC(12) + RC(13) + RC(14);
	short R2 = RC(20) + RC(21) + RC(22) + RC(23) + RC(24);
	short R3 = RC(30) + RC(31) + RC(32) + RC(33) + RC(34);
	short R4 = RC(40) + RC(41) + RC(42) + RC(43) + RC(44);

	short G0 = GC(00) + GC(01) + GC(02) + GC(03) + GC(04);
	short G1 = GC(10) + GC(11) + GC(12) + GC(13) + GC(14);
	short G2 = GC(20) + GC(21) + GC(22) + GC(23) + GC(24);
	short G3 = GC(30) + GC(31) + GC(32) + GC(33) + GC(34);
	short G4 = GC(40) + GC(41) + GC(42) + GC(43) + GC(44);

	short B0 = BC(00) + BC(01) + BC(02) + BC(03) + BC(04);
	short B1 = BC(10) + BC(11) + BC(12) + BC(13) + BC(14);
	short B2 = BC(20) + BC(21) + BC(22) + BC(23) + BC(24);
	short B3 = BC(30) + BC(31) + BC(32) + BC(33) + BC(34);
	short B4 = BC(40) + BC(41) + BC(42) + BC(43) + BC(44);


	dst[dst_index+0] = (R0 + R1 + R2 + R3 + R4 + 128) >> 8;
	dst[dst_index+1] = (G0 + G1 + G2 + G3 + G4 + 128) >> 8;
	dst[dst_index+2] = (B0 + B1 + B2 + B3 + B4 + 128) >> 8;
}



__global__ void PyrUp(unsigned char* input, unsigned char* src, unsigned char* dst, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int dst_index = (col + row * cols)*bpp;
    row = row << 1;
    col = col << 1;

    //int col0 = max(col - 2, 0);
    int col1 = max(col - 1, 0);
    int col2 = col;
    int col3 = min(col + 1, cols - 1);
    //int col4 = min(col + 2, cols - 1);

    //int row0 = max(row - 2, 0);
    int row1 = max(row - 1, 0);
    int row2 = row;
    int row3 = min(row + 1, rows - 1);
    //int row4 = min(row + 2, rows - 1);

    //unsigned char R00 = src[bpp*(row0*cols * 2 + col0)];
    //unsigned char R01 = src[bpp*(row0*cols * 2 + col1)];
    //unsigned char R02 = src[bpp*(row0*cols * 2 + col2)];
    //unsigned char R03 = src[bpp*(row0*cols * 2 + col3)];
    //unsigned char R04 = src[bpp*(row0*cols * 2 + col4)];

    //unsigned char R10 = src[bpp*(row1*cols * 2 + col0)];
    unsigned char R11 = src[bpp*(row1*cols * 2 + col1)];
    unsigned char R12 = src[bpp*(row1*cols * 2 + col2)];
    unsigned char R13 = src[bpp*(row1*cols * 2 + col3)];
    //unsigned char R14 = src[bpp*(row1*cols * 2 + col4)];

    //unsigned char R20 = src[bpp*(row2*cols * 2 + col0)];
    unsigned char R21 = src[bpp*(row2*cols * 2 + col1)];
    unsigned char R22 = src[bpp*(row2*cols * 2 + col2)];
    unsigned char R23 = src[bpp*(row2*cols * 2 + col3)];
    //unsigned char R24 = src[bpp*(row2*cols * 2 + col4)];

    //unsigned char R30 = src[bpp*(row3*cols * 2 + col0)];
    unsigned char R31 = src[bpp*(row3*cols * 2 + col1)];
    unsigned char R32 = src[bpp*(row3*cols * 2 + col2)];
    unsigned char R33 = src[bpp*(row3*cols * 2 + col3)];
    //unsigned char R34 = src[bpp*(row3*cols * 2 + col4)];

    //unsigned char R40 = src[bpp*(row4*cols * 2 + col0)];
    //unsigned char R41 = src[bpp*(row4*cols * 2 + col1)];
    //unsigned char R42 = src[bpp*(row4*cols * 2 + col2)];
    //unsigned char R43 = src[bpp*(row4*cols * 2 + col3)];
    //unsigned char R44 = src[bpp*(row4*cols * 2 + col4)];


    //unsigned char G00 = src[1 + bpp*(row0*cols * 2 + col0)];
    //unsigned char G01 = src[1 + bpp*(row0*cols * 2 + col1)];
    //unsigned char G02 = src[1 + bpp*(row0*cols * 2 + col2)];
    //unsigned char G03 = src[1 + bpp*(row0*cols * 2 + col3)];
    //unsigned char G04 = src[1 + bpp*(row0*cols * 2 + col4)];

    //unsigned char G10 = src[1 + bpp*(row1*cols * 2 + col0)];
    unsigned char G11 = src[1 + bpp*(row1*cols * 2 + col1)];
    unsigned char G12 = src[1 + bpp*(row1*cols * 2 + col2)];
    unsigned char G13 = src[1 + bpp*(row1*cols * 2 + col3)];
    //unsigned char G14 = src[1 + bpp*(row1*cols * 2 + col4)];

    //unsigned char G20 = src[1 + bpp*(row2*cols * 2 + col0)];
    unsigned char G21 = src[1 + bpp*(row2*cols * 2 + col1)];
    unsigned char G22 = src[1 + bpp*(row2*cols * 2 + col2)];
    unsigned char G23 = src[1 + bpp*(row2*cols * 2 + col3)];
    //unsigned char G24 = src[1 + bpp*(row2*cols * 2 + col4)];

    //unsigned char G30 = src[1 + bpp*(row3*cols * 2 + col0)];
    unsigned char G31 = src[1 + bpp*(row3*cols * 2 + col1)];
    unsigned char G32 = src[1 + bpp*(row3*cols * 2 + col2)];
    unsigned char G33 = src[1 + bpp*(row3*cols * 2 + col3)];
    //unsigned char G34 = src[1 + bpp*(row3*cols * 2 + col4)];

    //unsigned char G40 = src[1 + bpp*(row4*cols * 2 + col0)];
    //unsigned char G41 = src[1 + bpp*(row4*cols * 2 + col1)];
    //unsigned char G42 = src[1 + bpp*(row4*cols * 2 + col2)];
    //unsigned char G43 = src[1 + bpp*(row4*cols * 2 + col3)];
    //unsigned char G44 = src[1 + bpp*(row4*cols * 2 + col4)];


    //unsigned char B00 = src[2 + bpp*(row0*cols * 2 + col0)];
    //unsigned char B01 = src[2 + bpp*(row0*cols * 2 + col1)];
    //unsigned char B02 = src[2 + bpp*(row0*cols * 2 + col2)];
    //unsigned char B03 = src[2 + bpp*(row0*cols * 2 + col3)];
    //unsigned char B04 = src[2 + bpp*(row0*cols * 2 + col4)];

    //unsigned char B10 = src[2 + bpp*(row1*cols * 2 + col0)];
    unsigned char B11 = src[2 + bpp*(row1*cols * 2 + col1)];
    unsigned char B12 = src[2 + bpp*(row1*cols * 2 + col2)];
    unsigned char B13 = src[2 + bpp*(row1*cols * 2 + col3)];
    //unsigned char B14 = src[2 + bpp*(row1*cols * 2 + col4)];

    //unsigned char B20 = src[2 + bpp*(row2*cols * 2 + col0)];
    unsigned char B21 = src[2 + bpp*(row2*cols * 2 + col1)];
    unsigned char B22 = src[2 + bpp*(row2*cols * 2 + col2)];
    unsigned char B23 = src[2 + bpp*(row2*cols * 2 + col3)];
    //unsigned char B24 = src[2 + bpp*(row2*cols * 2 + col4)];

    //unsigned char B30 = src[2 + bpp*(row3*cols * 2 + col0)];
    unsigned char B31 = src[2 + bpp*(row3*cols * 2 + col1)];
    unsigned char B32 = src[2 + bpp*(row3*cols * 2 + col2)];
    unsigned char B33 = src[2 + bpp*(row3*cols * 2 + col3)];
    //unsigned char B34 = src[2 + bpp*(row3*cols * 2 + col4)];

    //unsigned char B40 = src[2 + bpp*(row4*cols * 2 + col0)];
    //unsigned char B41 = src[2 + bpp*(row4*cols * 2 + col1)];
    //unsigned char B42 = src[2 + bpp*(row4*cols * 2 + col2)];
    //unsigned char B43 = src[2 + bpp*(row4*cols * 2 + col3)];
    //unsigned char B44 = src[2 + bpp*(row4*cols * 2 + col4)];

# define RC(yx) R##yx*C##yx
# define GC(yx) G##yx*C##yx
# define BC(yx) B##yx*C##yx


    //short R0 = RC(00) + RC(01) + RC(02) + RC(03) + RC(04);
    short R1 = /*RC(10) +*/ RC(11) + RC(12) + RC(13)/*+ RC(14)*/;
    short R2 = /*RC(20) +*/ RC(21) + RC(22) + RC(23)/*+ RC(24)*/;
    short R3 = /*RC(30) +*/ RC(31) + RC(32) + RC(33)/*+ RC(34)*/;
    //short R4 = RC(40) + RC(41) + RC(42) + RC(43) + RC(44);

    //short G0 = GC(00) + GC(01) + GC(02) + GC(03) + GC(04);
    short G1 = /*GC(10) +*/ GC(11) + GC(12) + GC(13)/* + GC(14)*/;
    short G2 = /*GC(20) +*/ GC(21) + GC(22) + GC(23)/* + GC(24)*/;
    short G3 = /*GC(30) +*/ GC(31) + GC(32) + GC(33)/* + GC(34)*/;
    //short G4 = GC(40) + GC(41) + GC(42) + GC(43) + GC(44);

    //short B0 = BC(00) + BC(01) + BC(02) + BC(03) + BC(04);
    short B1 = /*BC(10)*/ + BC(11) + BC(12) + BC(13)/* + BC(14)*/;
    short B2 = /*BC(20)*/ + BC(21) + BC(22) + BC(23)/* + BC(24)*/;
    short B3 = /*BC(30)*/ + BC(31) + BC(32) + BC(33)/* + BC(34)*/;
    //short B4 = BC(40) + BC(41) + BC(42) + BC(43) + BC(44);


    //dst[dst_index + 0] = (R0 + R1 + R2 + R3 + R4 + 128) >> 8;
    //dst[dst_index + 1] = (G0 + G1 + G2 + G3 + G4 + 128) >> 8;
    //dst[dst_index + 2] = (B0 + B1 + B2 + B3 + B4 + 128) >> 8;

    short R = (R1 + R2 + R3 + 128 - input[dst_index + 0] + 128) >> 9;
    short G = (G1 + G2 + G3 + 128 - input[dst_index + 1] + 128) >> 9;
    short B = (B1 + B2 + B3 + 128 - input[dst_index + 2] + 128) >> 9;

    dst[dst_index + 0] = R;
    dst[dst_index + 1] = G;
    dst[dst_index + 2] = B;
}




# define C0 1
# define C1 4
# define C2 6
# define C3 4
# define C4 1

__global__ void BlurHorizontal(unsigned char* src, unsigned char* dst, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int dst_index = (col + row * cols/2)*bpp;
    //row = row << 1;
    col = col << 1;

    int dst_rows = rows;
    int dst_cols = cols;

    int src_rows = rows;
    int src_cols = cols;

    int col0 = max(col - 2, 0);
    int col1 = max(col - 1, 0);
    int col2 = col;
    int col3 = min(col + 1, cols - 1);
    int col4 = min(col + 2, cols - 1);

    unsigned char R0 = src[bpp*(row*src_cols + col0)];
    unsigned char R1 = src[bpp*(row*src_cols + col1)];
    unsigned char R2 = src[bpp*(row*src_cols + col2)];
    unsigned char R3 = src[bpp*(row*src_cols + col3)];
    unsigned char R4 = src[bpp*(row*src_cols + col4)];


    unsigned char G0 = src[1 + bpp*(row*src_cols + col0)];
    unsigned char G1 = src[1 + bpp*(row*src_cols + col1)];
    unsigned char G2 = src[1 + bpp*(row*src_cols + col2)];
    unsigned char G3 = src[1 + bpp*(row*src_cols + col3)];
    unsigned char G4 = src[1 + bpp*(row*src_cols + col4)];


    unsigned char B0 = src[2 + bpp*(row*src_cols + col0)];
    unsigned char B1 = src[2 + bpp*(row*src_cols + col1)];
    unsigned char B2 = src[2 + bpp*(row*src_cols + col2)];
    unsigned char B3 = src[2 + bpp*(row*src_cols + col3)];
    unsigned char B4 = src[2 + bpp*(row*src_cols + col4)];


    dst[dst_index + 0] = min(255, (R0*C0 + R1*C1 + R2*C2 + R3*C3 + R4*C4 + 8) >> 4);
    dst[dst_index + 1] = min(255, (G0*C0 + G1*C1 + G2*C2 + G3*C3 + G4*C4 + 8) >> 4);
    dst[dst_index + 2] = min(255, (B0*C0 + B1*C1 + B2*C2 + B3*C3 + B4*C4 + 8) >> 4);
}


__global__ void BlurVertical(unsigned char* src, unsigned char* dst, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int dst_index = (col + row * cols)*bpp;
    row = row << 1;
    //col = col << 1;

    //int dst_rows = rows;
    //int dst_cols = cols;

    //int src_rows = rows * 2;
    int src_cols = cols;

    int row0 = max(row - 2, 0);
    int row1 = max(row - 1, 0);
    int row2 = row;
    int row3 = min(row + 1, rows - 1);
    int row4 = min(row + 2, rows - 1);

    unsigned char R0 = src[bpp*(row0*src_cols + col)];
    unsigned char R1 = src[bpp*(row1*src_cols + col)];
    unsigned char R2 = src[bpp*(row2*src_cols + col)];
    unsigned char R3 = src[bpp*(row3*src_cols + col)];
    unsigned char R4 = src[bpp*(row4*src_cols + col)];


    unsigned char G0 = src[1 + bpp*(row0*src_cols + col)];
    unsigned char G1 = src[1 + bpp*(row1*src_cols + col)];
    unsigned char G2 = src[1 + bpp*(row2*src_cols + col)];
    unsigned char G3 = src[1 + bpp*(row3*src_cols + col)];
    unsigned char G4 = src[1 + bpp*(row4*src_cols + col)];


    unsigned char B0 = src[2 + bpp*(row0*src_cols + col)];
    unsigned char B1 = src[2 + bpp*(row1*src_cols + col)];
    unsigned char B2 = src[2 + bpp*(row2*src_cols + col)];
    unsigned char B3 = src[2 + bpp*(row3*src_cols + col)];
    unsigned char B4 = src[2 + bpp*(row4*src_cols + col)];


    dst[dst_index + 0] = min(255, (R0*C0 + R1*C1 + R2*C2 + R3*C3 + R4*C4 + 8) >> 4);
    dst[dst_index + 1] = min(255, (G0*C0 + G1*C1 + G2*C2 + G3*C3 + G4*C4 + 8) >> 4);
    dst[dst_index + 2] = min(255, (B0*C0 + B1*C1 + B2*C2 + B3*C3 + B4*C4 + 8) >> 4);
}



int main()
{

	int rows = 1300, cols = 1024, c = 3;
	ReadTGA("src.tga", cols, rows, c, img);

    dim3 threadsPerBlock(32, 32);
    dim3 threadsPerBlockH(32, 32); //(16,64 ) = 10+ ms, (32, 32) = 8+ms, (64, 16) = 10+ms
    dim3 threadsPerBlockV(512, 2);
    dim3 numBlocks((cols/2) / threadsPerBlock.x, (rows/2) / threadsPerBlock.y);
    dim3 numBlocksH((cols / 2) / threadsPerBlockH.x, (rows) / threadsPerBlockH.y);
    dim3 numBlocksV((cols / 2) / threadsPerBlockV.x, (rows / 2) / threadsPerBlockV.y);

    //cv::cuda::GpuMat gpuOut;
	g_time_logger.Start();
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
            //Blur << < numBlocks, threadsPerBlock >> > (img, out, rows/2, cols/2);
            BlurHorizontal << < numBlocksH, threadsPerBlockH >> > (img, outH, rows, cols);
            cudaDeviceSynchronize();
            LOG(h_blur);
            BlurVertical << < numBlocksV, threadsPerBlockV >> > (outH, outV, rows/2, cols/2);
            cudaDeviceSynchronize();
            LOG(v_blur);
            //PyrUp << < numBlocksV, threadsPerBlock >> > (img, outV, out, rows / 2, cols / 2);
            //cudaDeviceSynchronize();
            //LOG(pyr_up);

            //GpuMat(int rows, int cols, int type, void *data
            //cuda::GpuMat gpuIn(rows, cols, CV_8UC3, img);
            //cuda::pyrDown(gpuIn, gpuOut);
		}
	}

	g_time_logger.Display();

    DumpTga("blurH.tga", cols / 2, rows, bpp, outH);
    DumpTga("blurV.tga", cols / 2, rows / 2, bpp, outV);
    DumpTga("pyrUp.tga", cols, rows, bpp, out);
	//DumpTga("copy.tga", cols, rows, bpp, img);

	return 0;
}