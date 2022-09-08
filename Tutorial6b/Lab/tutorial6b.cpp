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
	std::vector<cl_int> inputVec1(16);
	std::vector<cl_int> inputVec2(16);
	std::vector<cl_int> results(32);
	cl::Buffer inputBuffer1, inputBuffer2, resultBuffer;

	// initialise values
	for (int i = 0; i < inputVec1.size(); i++)
	{
		inputVec1[i] = i;
		inputVec2[i] = i - 1;
		results[i] = -1;
	}

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
		if(!build_program(&program, &context, "vectors.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "vectors");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		inputBuffer1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * inputVec1.size(), &inputVec1[0]);
		inputBuffer2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * inputVec2.size(), &inputVec2[0]);
		resultBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * results.size());

		// set kernel arguments
		kernel.setArg(0, inputBuffer1);
		kernel.setArg(1, inputBuffer2);
		kernel.setArg(2, resultBuffer);

		// enqueue kernel for execution
		cl::NDRange offset(0);
		cl::NDRange globalSize(4);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(cl_int) * results.size(), &results[0]);

		// output result
		std::cout << "Results: " << std::endl;
		for (int i = 0; i < results.size(); i++) {
			std::cout << i << ": " << results[i] << std::endl;
		}
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
