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

#define LENGTH 1000

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_float> vectorA(LENGTH);
	std::vector<cl_float> vectorB(LENGTH);
	std::vector<cl_float> result(LENGTH);
	std::vector<cl_float> correctResult(LENGTH);

	cl::Buffer bufferA, bufferB, resultBuffer;

	// initialise values
	for (int i = 0; i < LENGTH; i++) 
	{
		// set vectors a and b to random values between 1.0 and 100.0
		vectorA[i] = 1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (100.0f - 1.0f)));
		vectorB[i] = 1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (100.0f - 1.0f)));
		result[i] = 0.0f;
		correctResult[i] = vectorA[i] + vectorB[i];
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
		if(!build_program(&program, &context, "vecadd.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "vecadd");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		bufferA = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * LENGTH, &vectorA[0]);
		bufferB = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * LENGTH, &vectorB[0]);
		resultBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * LENGTH);

		// set kernel arguments
		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferB);
		kernel.setArg(2, resultBuffer);

		// enqueue kernel for execution
		//queue.enqueueTask(kernel);

		cl::NDRange offset(0);
		cl::NDRange globalSize(LENGTH);	// work-units per kernel

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
			   
		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(cl_float) * LENGTH, &result[0]);

		// check the results
		cl_bool result_check = CL_TRUE;

		for (int i = 0; i < LENGTH; i++) 
		{
			if (result[i] != correctResult[i])
			{
				std::cout << "Error at result[" << i << "]: " << result[i] << std::endl;
				result_check = CL_FALSE;
			}
		}
		if (result_check)
		{
			std::cout << "Successfully processed " << LENGTH << " elements." << std::endl;
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
