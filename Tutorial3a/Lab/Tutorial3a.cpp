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

#define LENGTH 40

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	std::vector<cl_float> floatDataA(LENGTH);	// float data A
	std::vector<cl_float> floatDataB(LENGTH);	// float data B
	std::vector<cl_float> floatOutput(LENGTH);	// float output data
	std::vector<cl_int> intData(LENGTH);		// int data
	std::vector<cl_int> intOutput(LENGTH);		// int output data

	cl::Buffer floatBufferA;
	cl::Buffer floatBufferB;
	cl::Buffer intBuffer;

	for (int i = 0; i < LENGTH; i++) {
		floatDataA[i] = i + 1.0f;			// set values from 1.0 to LENGTH.0
		floatDataB[i] = -1.0f * (i + 1);	// set values from -1.0 to -LENGTH.0
		intData[i] = i;						// set values from 0 to LENGTH-1
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
		if(!build_program(&program, &context, "blank_kernel.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "blank");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		floatBufferA = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * floatDataA.size(), &floatDataA[0]);
		floatBufferB = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * LENGTH);
		intBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * intData.size(), &intData[0]);

		// output buffer sizes and locations
		std::cout << "floatBufferA size: " << floatBufferA.getInfo<CL_MEM_SIZE>() << std::endl;
		std::cout << "floatBufferB size: " << floatBufferB.getInfo<CL_MEM_SIZE>() << std::endl;
		std::cout << "intBuffer size: " << intBuffer.getInfo<CL_MEM_SIZE>() << std::endl;
		std::cout << "floatDataA memory location: " << &floatDataA[0] << std::endl;
		std::cout << "floatBufferA memory location: " << floatBufferA.getInfo<CL_MEM_HOST_PTR>() << std::endl;
		std::cout << "floatDataB memory location: " << &floatDataB[0] << std::endl;
		std::cout << "floatBufferB memory location: " << floatBufferB.getInfo<CL_MEM_HOST_PTR>() << std::endl;
		std::cout << "intData memory location: " << &intData[0] << std::endl;
		std::cout << "intBuffer memory location: " << intBuffer.getInfo<CL_MEM_HOST_PTR>() << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to write from host to device memory
		queue.enqueueWriteBuffer(floatBufferB, CL_TRUE, 0, sizeof(cl_float) * floatDataB.size(), &floatDataB[0]);
		
		// set kernel arguments
		int a = 0;
		kernel.setArg(0, a);
		kernel.setArg(1, floatBufferA);
		kernel.setArg(2, floatBufferB);
		kernel.setArg(3, intBuffer);

		// enqueue kernel for execution
		queue.enqueueTask(kernel);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(floatBufferA, CL_TRUE, 0, sizeof(cl_float) * LENGTH, &floatOutput[0]);

		// output contents
		std::cout << "\nContents of floatBufferA: " << std::endl;

		for (int i = 0; i < LENGTH / 10; i++)
		{
			for (int j = 0; j < 10; j++) 
			{
				std::cout << floatOutput[j + i * 10] << " ";
			}
			std::cout << std::endl;
		}

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(floatBufferB, CL_TRUE, 0, sizeof(cl_float) * LENGTH, &floatOutput[0]);

		// output contents
		std::cout << "\nContents of floatBufferB: " << std::endl;

		for (int i = 0; i < LENGTH / 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				std::cout << floatOutput[j + i * 10] << " ";
			}
			std::cout << std::endl;
		}

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(intBuffer, CL_TRUE, 0, sizeof(cl_int) * LENGTH, &intOutput[0]);

		// output contents
		std::cout << "\nContents of intBuffer: " << std::endl;

		for (int i = 0; i < LENGTH / 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				std::cout << intOutput[j + i * 10] << " ";
			}
			std::cout << std::endl;
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

