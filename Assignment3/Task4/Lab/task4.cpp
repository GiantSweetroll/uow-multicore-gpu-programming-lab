#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "common.h"
#include "bmpfuncs.h"

#define NUM_ITERATIONS 1000

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
	unsigned char* inputImageLum;
	unsigned char* inputImageBlurHorz;
	unsigned char* inputImageBlurBoth;
	unsigned char* outputImageLum;
	unsigned char* outputImageBlur;
	unsigned char* outputImage;
	int imgWidth, imgHeight, imageSize;
	float lum_t;

	cl::ImageFormat imgFormat;
	cl::Image2D inputImgBuffer, inputImgBufferLum, inputImgBufferBlurHorz, inputImgBufferBlurBoth, outputImgBufferLum, outputImgBufferBlur, outputImgBuffer;

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
		if(!build_program(&program, &context, "task4.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "glowing_pixels");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// read user's luminance threshold value
		std::cout << "Please enter a threshold value for luminance (0.0 - 1.0): ";
		std::cin >> lum_t;
		std::cout << std::endl;

		if (lum_t < 0.0f || lum_t > 1.0f) {
			quit_program("Invalid luminance range");
		}
		
		// read input image
		inputImage = read_BMP_RGB_to_RGBA("peppers.bmp", &imgWidth, &imgHeight);

		// allocate memory for output image
		imageSize = imgWidth * imgHeight * 4;
		outputImageLum = new unsigned char[imageSize];
		outputImageBlur = new unsigned char[imageSize];
		outputImage = new unsigned char[imageSize];

		// image format
		imgFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);

		// create image objects
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBufferLum = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImageLum);
		outputImgBufferBlur = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImageBlur);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		// set kernel arguments
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, lum_t);
		kernel.setArg(2, outputImgBufferLum);
		
		// enqueue kernel for horizontal pass
		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imgWidth, imgHeight);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Glowing Pixels Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read image from device to host memory
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imgWidth;
		region[1] = imgHeight;
		region[2] = 1;

		// enqueue command to read image from device to host memory
		queue.enqueueReadImage(outputImgBufferLum, CL_TRUE, origin, region, 0, 0, outputImageLum);

		// output results to image file
		write_BMP_RGBA_to_RGB("Task4a.bmp", outputImageLum, imgWidth, imgHeight);

		// read input image (lum)
		inputImageLum = read_BMP_RGB_to_RGBA("Task4a.bmp", &imgWidth, &imgHeight);
		inputImgBufferLum = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImageLum);

		// set kernel for blurring task
		kernel = cl::Kernel(program, "blur_pass");

		// set kernel arguments for horizontal pass
		kernel.setArg(0, inputImgBufferLum);
		kernel.setArg(1, outputImgBufferBlur);
		kernel.setArg(2, 0);

		// enqueue kernel for horizontal pass
		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Horizontal pass blurring Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read image from device to host memory
		queue.enqueueReadImage(outputImgBufferBlur, CL_TRUE, origin, region, 0, 0, outputImageBlur);

		// output results to image file
		write_BMP_RGBA_to_RGB("Task4b.bmp", outputImageBlur, imgWidth, imgHeight);

		// read input image (BlurHorz)
		inputImageBlurHorz = read_BMP_RGB_to_RGBA("Task4b.bmp", &imgWidth, &imgHeight);
		inputImgBufferBlurHorz = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImageBlurHorz);

		// set kernel arguments for vertical pass
		kernel.setArg(0, inputImgBufferBlurHorz);
		kernel.setArg(1, outputImgBufferBlur);
		kernel.setArg(2, 1);

		// enqueue kernel for vertical pass
		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Vertical pass blurring Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read image from device to host memory
		queue.enqueueReadImage(outputImgBufferBlur, CL_TRUE, origin, region, 0, 0, outputImageBlur);

		// output results to image file
		write_BMP_RGBA_to_RGB("Task4c.bmp", outputImageBlur, imgWidth, imgHeight);

		// read input image (BlurBoth)
		inputImageBlurBoth = read_BMP_RGB_to_RGBA("Task4c.bmp", &imgWidth, &imgHeight);
		inputImgBufferBlurBoth = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImageBlurBoth);

		// set kernel for bloom effect
		kernel = cl::Kernel(program, "bloom");

		// set kernel arguments for bloom effect
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, inputImgBufferBlurBoth);
		kernel.setArg(2, outputImgBuffer);

		// enqueue kernel for bloom
		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Bloom Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read image from device to host memory
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		write_BMP_RGBA_to_RGB("Task4d.bmp", outputImage, imgWidth, imgHeight);

		std::cout << "Done." << std::endl;

		// deallocate memory
		free(inputImage);
		free(outputImageLum);
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
