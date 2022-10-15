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

#define NUM_INTS 4096
#define NUM_ITEMS 512
#define NUM_ITERATIONS 2000

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_int> data(NUM_INTS);
	cl::Buffer dataBuffer;
	cl_int numOfInts = NUM_INTS;

	// initialise data
	for (int i = 0; i < NUM_INTS; i++) {
		data[i] = i;
	}

	// declare events
	cl::Event profileEvent;
	cl_ulong timeStart, timeEnd, timeTotal;

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
		if(!build_program(&program, &context, "profile_items.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "profile_items");

		// create command queue
		queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		// create buffers
		dataBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * NUM_INTS);

		// set kernel arguments
		kernel.setArg(0, dataBuffer);
		kernel.setArg(1, numOfInts);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		cl::NDRange offset(0);
		cl::NDRange globalSize(NUM_ITEMS);
		timeTotal = 0;

		for (int i = 0; i < NUM_ITERATIONS; i++)
		{
			// enqueue kernel for execution
			queue.enqueueNDRangeKernel(kernel, offset, globalSize, cl::NullRange, NULL, &profileEvent);
			queue.finish();

			timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

			timeTotal += timeEnd - timeStart;
		}

		printf("Average time = %lu\n", timeTotal / NUM_ITERATIONS);
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
