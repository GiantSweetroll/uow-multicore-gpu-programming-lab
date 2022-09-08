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

#define OFFSET 8
#define GLOBAL_SIZE 16
#define LOCAL_SIZE 4

#define OUTPUT_SIZE (GLOBAL_SIZE + 1) * 4

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_int> inputVec(1000);
	std::vector<cl_int> outputVec(OUTPUT_SIZE);

	cl::Buffer inputBuffer, outputBuffer;

	// initialise values
	for (int i = 0; i < 1000; i++)
	{
		inputVec[i] = 1000 + i;
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
		if(!build_program(&program, &context, "workitem_id.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "workitem_id");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * inputVec.size(), &inputVec[0]);
		outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * outputVec.size());

		// set kernel arguments
		kernel.setArg(0, inputBuffer);
		kernel.setArg(1, outputBuffer);

		// enqueue kernel for execution
		cl::NDRange offset(OFFSET);
		cl::NDRange globalSize(GLOBAL_SIZE);
		cl::NDRange localSize(LOCAL_SIZE);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(cl_int) * outputVec.size(), &outputVec[0]);

		// output results
		std::cout << "\nNumber of work-items: " << outputVec[0] << std::endl;
		std::cout << "Work-item offset: " << outputVec[1] << std::endl;
		std::cout << "Number of work-groups: " << outputVec[2] << std::endl;
		std::cout << "Work-items per group: " << outputVec[3] << std::endl;

		for (int i = 1, index = 4; i <= GLOBAL_SIZE; i++, index += 4) 
		{
			std::cout << "\nWork-item " << i << std::endl;
			std::cout << "Global ID: " << outputVec[index];
			std::cout << "\tWork-group ID: " << outputVec[index + 1];
			std::cout << "\tLocal ID: " << outputVec[index + 2];
			std::cout << "\tOutput: " << outputVec[index + 3] << std::endl;
			std::cout << "--------------------";
		}
		std::cout << std::endl;
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
