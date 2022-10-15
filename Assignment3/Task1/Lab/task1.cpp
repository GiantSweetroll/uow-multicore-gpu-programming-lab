#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "common.h"
#include "bmpfuncs.h"

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	unsigned char* inputImage;
	unsigned char* outputImageHorz;
	unsigned char* outputImageVert;
	unsigned char* outputImageBoth;
	int imgWidth, imgHeight, imageSize;

	cl::ImageFormat imgFormat;
	cl::Image2D inputImgBuffer, outputImgBufferHorz, outputImgBufferVert, outputImgBufferBoth;

	try {
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program("Device not selected.");
		}

		// create a context from device
		context = cl::Context(device);

		// build the program
		if(!build_program(&program, &context, "task1.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "task1");

		// create command queue
		queue = cl::CommandQueue(context, device);
		
		// read input image
		inputImage = read_BMP_RGB_to_RGBA("peppers.bmp", &imgWidth, &imgHeight);

		// allocate memory for output image
		imageSize = imgWidth * imgHeight * 4;
		outputImageHorz = new unsigned char[imageSize];
		outputImageVert = new unsigned char[imageSize];
		outputImageBoth = new unsigned char[imageSize];

		// image format
		imgFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);

		// create image objects
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBufferHorz = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImageHorz);
		outputImgBufferVert = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImageVert);
		outputImgBufferBoth = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImageBoth);

		// set kernel arguments
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBufferHorz);
		kernel.setArg(2, outputImgBufferVert);
		kernel.setArg(3, outputImgBufferBoth);

		// enqueue kernel
		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imgWidth, imgHeight);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read image from device to host memory
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imgWidth;
		region[1] = imgHeight;
		region[2] = 1;

		queue.enqueueReadImage(outputImgBufferHorz, CL_TRUE, origin, region, 0, 0, outputImageHorz);
		queue.enqueueReadImage(outputImgBufferVert, CL_TRUE, origin, region, 0, 0, outputImageVert);
		queue.enqueueReadImage(outputImgBufferBoth, CL_TRUE, origin, region, 0, 0, outputImageBoth);

		// output results to image file
		write_BMP_RGBA_to_RGB("Task1a.bmp", outputImageHorz, imgWidth, imgHeight);
		write_BMP_RGBA_to_RGB("Task1b.bmp", outputImageVert, imgWidth, imgHeight);
		write_BMP_RGBA_to_RGB("Task1c.bmp", outputImageBoth, imgWidth, imgHeight);

		std::cout << "Done." << std::endl;

		// deallocate memory
		free(inputImage);
		free(outputImageHorz);
		free(outputImageVert);
		free(outputImageBoth);
	}
	// catch any OpenCL function errors
	catch (cl::Error e) {
		// call function to handle errors
		handle_error(e);
	}

#ifdef _WIN32
	// wait for a keypress on Windows OS before exiting
	std::cout << "\npress a key to quit...";
	std::cin.ignore();
#endif

	return 0;
}
