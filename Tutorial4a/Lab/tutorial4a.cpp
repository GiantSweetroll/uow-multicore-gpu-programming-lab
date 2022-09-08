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

#define OFFSET 0
#define NUM_OF_WORK_ITEMS 10

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	std::vector<cl_int> result(NUM_OF_WORK_ITEMS);
	cl::Buffer outputBuffer;

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
		if(!build_program(&program, &context, "global_id.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "global_id");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * NUM_OF_WORK_ITEMS);

		// set kernel arguments
		kernel.setArg(0, outputBuffer);

		// enqueue kernel for execution
		//queue.enqueueTask(kernel);

		cl::NDRange offset(OFFSET);
		cl::NDRange globalSize(NUM_OF_WORK_ITEMS);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
			   
		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(cl_float) * NUM_OF_WORK_ITEMS, &result[0]);

		// check the results
		std::cout << "--------------------" << std::endl;
		std::cout << "Work-item global IDs" << std::endl;

		for (int i = 0; i < NUM_OF_WORK_ITEMS; i++)
		{
			std::cout << "Work-item " << i << "'s global ID: " << result[i] << std::endl;
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
