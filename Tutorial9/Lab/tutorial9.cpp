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

#define TEXT_FILENAME "kafka.txt"

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	cl::Buffer textBuffer, resultBuffer;	// input and output buffers
	cl::LocalSpaceArg localSpace;			// to create local space for the kernel
	cl_char pattern[16] = { 't', 'h', 'a', 't', 'w', 'i', 't', 'h', 'h', 'a', 'v', 'e', 'f', 'r', 'o', 'm' };
	cl_int result[4] = { 0, 0, 0, 0 };
	cl_int charsPerWorkitem = 0, workgroupSize;

	// open input file stream to text file
	std::ifstream textFile(TEXT_FILENAME);

	// check whether file was opened
	if (!textFile.is_open())
	{
		std::cout << "File not found." << std::endl;
		return false;
	}

	// create text string and load contents from the file
	std::string text(std::istreambuf_iterator<char>(textFile), (std::istreambuf_iterator<char>()));

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
		if(!build_program(&program, &context, "string_search.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "string_search");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// characters per work-item
		workgroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		charsPerWorkitem = (text.size() / workgroupSize) + 1;
		text.resize((charsPerWorkitem + 1) * workgroupSize);

		// create buffers
		textBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_char) * text.size(), &text[0]);
		resultBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(result), &result[0]);
		localSpace = cl::Local(sizeof(result));

		// set kernel arguments
		kernel.setArg(0, sizeof(pattern), pattern);
		kernel.setArg(1, textBuffer);
		kernel.setArg(2, charsPerWorkitem);
		kernel.setArg(3, localSpace);
		kernel.setArg(4, resultBuffer);

		// enqueue kernel for execution
		cl::NDRange offset(0);
		cl::NDRange globalSize(workgroupSize);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
			   
		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(result), &result[0]);

		std::cout << "\nResults:" << std::endl;
		std::cout << "Number of occurrences of 'that': " << result[0] << std::endl;
		std::cout << "Number of occurrences of 'with': " << result[1] << std::endl;
		std::cout << "Number of occurrences of 'have': " << result[2] << std::endl;
		std::cout << "Number of occurrences of 'from': " << result[3] << std::endl;
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
