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
	std::vector<cl_float> result(24);
	cl::Buffer resultBuffer;

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
		if(!build_program(&program, &context, "id_check.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "id_check");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		resultBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * 24);

		// set kernel arguments
		kernel.setArg(0, resultBuffer);

		// enqueue kernel for execution
		cl::NDRange offset(3, 5);
		cl::NDRange globalSize(6, 4);
		cl::NDRange localSize(3, 2);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(cl_float) * 24, &result[0]);

		// output result
		for (int i = 0; i < 24; i += 6) 
		{
			printf("%.2f     %.2f     %.2f     %.2f     %.2f     %.2f\n",
				result[i], result[i + 1], result[i + 2], result[i + 3], result[i + 4], result[i + 5]);
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
