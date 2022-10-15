#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "common.h"

#define NUM_OF_ELEMENTS 131072

enum Kernels {SCALAR, VECTOR};

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel[2];			// kernel objects
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_float> data(NUM_OF_ELEMENTS), scalarSum, vectorSum;
	cl::Buffer dataBuffer, scalarBuffer, vectorBuffer;	 
	cl::LocalSpaceArg localSpace;				// to create local space for the kernel
	cl::Event profileEvent;						// for profiling
	cl_ulong timeStart, timeEnd, timeTotal;
	cl_int numOfGroups;							// number of work-groups
	cl_float sum, correctSum;					// results
	size_t workgroupSize;						// work group size
    size_t kernelWorkgroupSize;                 // allowed work group size for the kernel
	cl_ulong localMemorySize;					// device's local memory size

	// initialise values
	for (int i = 0; i < NUM_OF_ELEMENTS; i++) {
		data[i] = 1.0f*i;
	}
	// calculate the correct result
	correctSum = 1.0f * NUM_OF_ELEMENTS / 2 * (NUM_OF_ELEMENTS - 1);

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
		if(!build_program(&program, &context, "reduction.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel[SCALAR] = cl::Kernel(program, "reduction_scalar");
		kernel[VECTOR] = cl::Kernel(program, "reduction_vector");

		// create command queue
		queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		// get device information
		workgroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		localMemorySize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
        kernelWorkgroupSize = kernel[SCALAR].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

		// display the information
		std::cout << "Max workgroup size: " << workgroupSize << std::endl;
		std::cout << "Local memory size: " << localMemorySize << std::endl;
        std::cout << "Kernel workgroup size: " << kernelWorkgroupSize << std::endl;

        // if kernel only allows one work-item per work-group, abort
        if (kernelWorkgroupSize == 1)
            quit_program("Abort: Cannot run reduction kernel, because kernel workgroup size is 1.");
        
        // if allowed kernel work group size smaller than device's max workgroup size
        if (workgroupSize > kernelWorkgroupSize)
            workgroupSize = kernelWorkgroupSize;

        // ensure sufficient local memory is available
		while (localMemorySize < sizeof(float) * workgroupSize * 4)
		{
			workgroupSize /= 4;
		}

		// compute number of groups and resize vectors
		numOfGroups = NUM_OF_ELEMENTS / workgroupSize;
		scalarSum.resize(numOfGroups);
		vectorSum.resize(numOfGroups/4);

		for (int i = 0; i < numOfGroups; i++)
		{
			scalarSum[i] = 0.0f;
		}
		for (int i = 0; i < numOfGroups/4; i++)
		{
			vectorSum[i] = 0.0f;
		}

		// create buffers
		dataBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * NUM_OF_ELEMENTS, &data[0]);
		scalarBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * numOfGroups, &scalarSum[0]);
		vectorBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * numOfGroups/4, &vectorSum[0]);

		cl::NDRange offset(0);
		cl::NDRange globalSize(0);
		cl::NDRange localSize(workgroupSize);

		for (int i = 0; i < 2; i++)
		{
			// set kernel arguments
			kernel[i].setArg(0, dataBuffer);

			if (i == SCALAR)
			{
				localSpace = cl::Local(sizeof(float) * workgroupSize);
				kernel[i].setArg(1, localSpace);
				kernel[i].setArg(2, scalarBuffer);
				globalSize = NUM_OF_ELEMENTS;
			}
			else
			{
				localSpace = cl::Local(sizeof(float) * workgroupSize * 4);
				kernel[i].setArg(1, localSpace);
				kernel[i].setArg(2, vectorBuffer);
				globalSize = NUM_OF_ELEMENTS/4;
			}

			// enqueue kernel for execution
			queue.enqueueNDRangeKernel(kernel[i], offset, globalSize, localSize, NULL, &profileEvent);

			std::cout << "Kernel enqueued." << std::endl;
			std::cout << "--------------------" << std::endl;

			queue.finish();

			// check timing
			timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

			timeTotal = timeEnd - timeStart;

			// read and check results
			if (i == SCALAR)
			{
				// enqueue command to read from device to host memory
				queue.enqueueReadBuffer(scalarBuffer, CL_TRUE, 0, sizeof(cl_float) * numOfGroups, &scalarSum[0]);

				sum = 0.0f;
				for (int i = 0; i < numOfGroups; i++)
				{
					sum += scalarSum[i];
				}

				std::cout << "Scalar reduction: ";
			}
			else
			{
				// enqueue command to read from device to host memory
				queue.enqueueReadBuffer(vectorBuffer, CL_TRUE, 0, sizeof(cl_float) * numOfGroups / 4, &vectorSum[0]);

				sum = 0.0f;
				for (int i = 0; i < numOfGroups / 4; i++)
				{
					sum += vectorSum[i];
				}

				std::cout << "Vector reduction: ";
			}

			// check results
			if (fabs(sum - correctSum) > 0.01f * fabs(sum))
			{
				std::cout << "Check failed." << std::endl;
			}
			else
			{
				std::cout << "Check passed." << std::endl;
			}

			std::cout << "Total time: " << timeTotal << std::endl;
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
