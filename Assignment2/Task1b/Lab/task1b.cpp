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

#define NUM_OF_WORK_ITEMS 4

int main(void)
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_int> vector1(32), vector2(16), results(8 * 4 * NUM_OF_WORK_ITEMS);	// for results, the size is determined by four int8 vectors * number of work items
	cl::Buffer vector1Buffer, vector2Buffer, outputBuffer;

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
		if (!build_program(&program, &context, "task1b.cl"))
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// initialize vector1 values
		int max = 9;
		int min = 1;
		for (int i = 0; i < vector1.capacity(); i++)
		{
			int num = rand() % (max - min + 1) + min;		// generate random number between 1 - 9 (inclusive)
			vector1[i] = num;
		}

		// initialize vector 2 values
		// insert number 2 - 9 in order for the first half
		int half = vector2.capacity() / 2;
		for (int i = 0; i < half; i++)
		{
			vector2[i] = i + 2;
		}
		// insert number -9 to -2 in sequence for second half
		for (int i = half; i < vector2.capacity(); i++)
		{
			vector2[i] = i - half - 9;
		}
		
		// create a kernel
		kernel = cl::Kernel(program, "task1b");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		vector1Buffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * vector1.capacity(), &vector1[0]);
		vector2Buffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * vector2.capacity(), &vector2[0]);
		outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * results.capacity());

		// set kernel arguments
		kernel.setArg(0, vector1Buffer);
		kernel.setArg(1, vector2Buffer);
		kernel.setArg(2, outputBuffer);

		// enqueue kernel for execution
		cl::NDRange offset(0);
		cl::NDRange globalSize(NUM_OF_WORK_ITEMS);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(cl_int) * NUM_OF_WORK_ITEMS * 32, &results[0]);

		// check the results
		std::cout << "--------------------" << std::endl;
		for (int i = 0; i < NUM_OF_WORK_ITEMS; i++)
		{
			int startingIndex = i * (results.size() / NUM_OF_WORK_ITEMS);	// get starting index from 1D array

			std::cout << "Work-item " << (i + 1) << std::endl;

			// print contents of v
			std::cout << "v	: ";
			for (int a = startingIndex; a < startingIndex + 8; a++)
			{
				std::cout << results[a] << " ";
			}
			std::cout << std::endl;

			// print contents of v1
			std::cout << "v1	: ";
			startingIndex += 8;
			for (int a = startingIndex; a < startingIndex + 8; a++)
			{
				std::cout << results[a] << " ";
			}
			std::cout << std::endl;

			// print contents of v2
			std::cout << "v2	: ";
			startingIndex += 8;
			for (int a = startingIndex; a < startingIndex + 8; a++)
			{
				std::cout << results[a] << " ";
			}
			std::cout << std::endl;

			// print contents of results
			std::cout << "results	: ";
			startingIndex += 8;
			for (int a = startingIndex; a < startingIndex + 8; a++)
			{
				std::cout << results[a] << " ";
			}
			std::cout << std::endl << std::endl;
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
