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

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_float> shuffle1(8);
	std::vector<cl_char> shuffle2(16);
	std::vector<cl_float> select1(4);
	std::vector<cl_uchar> select2(2);
	cl::Buffer resultBuffer1, resultBuffer2, resultBuffer3, resultBuffer4;

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
		if(!build_program(&program, &context, "shuffle_select.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "shuffle_select");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		resultBuffer1 = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * 8);
		resultBuffer2 = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_char) * 16);
		resultBuffer3 = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * 4);
		resultBuffer4 = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uchar) * 2);

		// set kernel arguments
		kernel.setArg(0, resultBuffer1);
		kernel.setArg(1, resultBuffer2);
		kernel.setArg(2, resultBuffer3);
		kernel.setArg(3, resultBuffer4);

		// enqueue kernel for execution
		queue.enqueueTask(kernel);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(resultBuffer1, CL_TRUE, 0, sizeof(cl_float) * 8, &shuffle1[0]);
		queue.enqueueReadBuffer(resultBuffer2, CL_TRUE, 0, sizeof(cl_char) * 16, &shuffle2[0]);
		queue.enqueueReadBuffer(resultBuffer3, CL_TRUE, 0, sizeof(cl_float) * 4, &select1[0]);
		queue.enqueueReadBuffer(resultBuffer4, CL_TRUE, 0, sizeof(cl_uchar) * 2, &select2[0]);

		// output results
		std::cout << "shuffle: ";
		for (int i = 0; i < 8; i++) {
			std::cout << shuffle1[i] << " ";
		}
		std::cout << std::endl << std::endl;

		std::cout << "shuffle2: ";
		for (int i = 0; i < 16; i++) {
			std::cout << shuffle2[i] << " ";
		}
		std::cout << std::endl << std::endl;

		std::cout << "select: ";
		for (int i = 0; i < 4; i++) {
			std::cout << select1[i] << " ";
		}
		std::cout << std::endl << std::endl;

		printf("bitselect: %X, %X\n", select2[0], select2[1]);
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
