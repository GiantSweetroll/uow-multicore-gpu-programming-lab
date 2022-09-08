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

#define GLOBAL_SIZE 16
#define LOCAL_SIZE 8
#define NUM_WORK_GROUPS GLOBAL_SIZE/LOCAL_SIZE

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_float> inputVec(1000);
	std::vector<cl_int> outputVec1(GLOBAL_SIZE);
	std::vector<cl_float> outputVec2(NUM_WORK_GROUPS);

	cl::Buffer inputBuffer, outputBuffer1, outputBuffer2;

	// initialise values
	for (int i = 0; i < 1000; i++)
	{
		inputVec[i] = 0.1f * i;
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
		if(!build_program(&program, &context, "memory.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "memory_spaces");

		// create command queue
		queue = cl::CommandQueue(context, device);

#ifdef __APPLE__
        // MacOS: cannot run kernel using CPU
        size_t kernelWorkgroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

        // abort if kernel only allows one work-item per work-group
        if (kernelWorkgroupSize == 1)
            quit_program("Abort: Cannot run kernel, because kernel workgroup size is 1.");
#endif

		// create buffers
		inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * inputVec.size(), &inputVec[0]);
		outputBuffer1 = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * outputVec1.size());
		outputBuffer2 = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * outputVec2.size());

		// set kernel arguments
		int a = 101;
		float b = 11.22;
		cl::LocalSpaceArg localSpace = cl::Local(sizeof(float) * LOCAL_SIZE);

		kernel.setArg(0, a);
		kernel.setArg(1, b);
		kernel.setArg(2, localSpace);
		kernel.setArg(3, inputBuffer);
		kernel.setArg(4, outputBuffer1);
		kernel.setArg(5, outputBuffer2);

		// enqueue kernel for execution
		cl::NDRange offset(0);
		cl::NDRange globalSize(GLOBAL_SIZE);
		cl::NDRange localSize(LOCAL_SIZE);

        queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(outputBuffer1, CL_TRUE, 0, sizeof(cl_int) * outputVec1.size(), &outputVec1[0]);
		queue.enqueueReadBuffer(outputBuffer2, CL_TRUE, 0, sizeof(cl_float) * outputVec2.size(), &outputVec2[0]);

		// output results
		for (int i = 0; i < GLOBAL_SIZE; i++) 
		{
			std::cout << "Work-item " << i << "\tOutput: " << outputVec1[i] << std::endl;
		}

		std::cout << std::endl;

		for (int i = 0; i < NUM_WORK_GROUPS; i++)
		{
			float resultCheck = 0;

			// check the result
			for (int j = 0; j < LOCAL_SIZE; j++)
			{
				resultCheck += inputVec[i * LOCAL_SIZE + j];
			}
			resultCheck += b;

			// check whether output matches the correct result
			if(resultCheck == outputVec2[i])
				std::cout << "Work-group " << i << "\tOutput: " << outputVec2[i] << std::endl;
			else
				std::cout << "Work-group " << i << ": Incorrect Result!" << std::endl;
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
