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
		floatDataA[i] = i + 1.0f;			// set values from 1.0 to 40.0
		floatDataB[i] = -1.0f * (i + 1);	// set values from -1.0 to -40.0
		intData[i] = i;						// set values from 0 to 39
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
		floatBufferA = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * floatDataA.size(), &floatDataA[0]);
		floatBufferB = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * floatDataB.size(), &floatDataB[0]);
		intBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * intData.size(), &intData[0]);
		
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
		queue.enqueueReadBuffer(floatBufferB, CL_TRUE, 0, sizeof(cl_float) * LENGTH, &floatOutput[0]);

		// output contents
		std::cout << "\nContents of floatBufferB BEFORE copy command: " << std::endl;

		for (int i = 0; i < LENGTH / 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				std::cout << floatOutput[j + i * 10] << " ";
			}
			std::cout << std::endl;
		}

		// enqueue command to copy contents from floatBufferA to floatBufferB
		queue.enqueueCopyBuffer(floatBufferA, floatBufferB, 0, 0, sizeof(cl_float) * LENGTH);

		// enqueue command to map buffer to host memory
		void* mappedMemory = queue.enqueueMapBuffer(floatBufferB, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_float) * LENGTH);

		// copy data from mapped memory
		memcpy(&floatOutput[0], mappedMemory, sizeof(cl_float) * LENGTH);
		
		// unmap the buffer
		queue.enqueueUnmapMemObject(floatBufferB, mappedMemory);

		// output contents
		std::cout << "\nContents of floatBufferB AFTER copy command: " << std::endl;

		for (int i = 0; i < LENGTH / 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				std::cout << floatOutput[j + i * 10] << " ";
			}
			std::cout << std::endl;
		}
		
		// read a rectangle of data from the buffer
		// set all values to 0
		memset(&intOutput[0], 0, sizeof(cl_int)*LENGTH);

		cl::size_t<3> bufferOrigin, hostOrigin, region;

		bufferOrigin[0] = 5 * sizeof(cl_int);
		bufferOrigin[1] = 2;
		bufferOrigin[2] = 0;
		hostOrigin[0] = 1 * sizeof(cl_int);
		hostOrigin[1] = 1;
		hostOrigin[2] = 0;
		region[0] = 4 * sizeof(cl_int);
		region[1] = 2;
		region[2] = 1;

		// enqueue command to read a rectangular section of the buffer to host memory
		queue.enqueueReadBufferRect(intBuffer, CL_TRUE, bufferOrigin, hostOrigin, region,
			10 * sizeof(cl_int), 0, 10 * sizeof(cl_int), 0, &intOutput[0]);

		// output contents of intData
		std::cout << "\nContents of intData: " << std::endl;

		for (int i = 0; i < LENGTH / 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				std::cout << intData[j + i * 10] << " ";
			}
			std::cout << std::endl;
		}

		// output contents
		std::cout << "\nContents of intOutput after read buffer rect command: " << std::endl;

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
