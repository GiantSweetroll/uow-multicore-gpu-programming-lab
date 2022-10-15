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

// callback function
void CL_CALLBACK read_complete(cl_event e, cl_int status, void* data) 
{
	// display the data
	float *float_data = (float*)data;
	printf("New data: %4.2f, %4.2f, %4.2f, %4.2f\n", float_data[0], float_data[1], float_data[2], float_data[3]);
}

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_float> data(4);
	cl::Buffer dataBuffer;

	// declare events
	cl::Event kernelEvent, readEvent;
	cl::UserEvent userEvent;
	std::vector<cl::Event> waitList[2];

	// initialise data
	for (int i = 0; i < 4; i++)
		data[i] = i * 1.0f;

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
		if(!build_program(&program, &context, "user_event.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "user_event");

		// create command queue
		if ((device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>() & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0)
		{
			std::cout << "CommandQueue: out-of-order execution supported." << std::endl;
			queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		}
		else
		{
			std::cout << "CommandQueue: out-of-order execution NOT supported." << std::endl;
			queue = cl::CommandQueue(context, device);
		}

		// create buffers
		dataBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 4, &data[0]);

		// set kernel arguments
		kernel.setArg(0, dataBuffer);

		// create user event
		userEvent = cl::UserEvent(context);
		waitList[0].push_back((cl::Event)userEvent);

		// enqueue kernel for execution
		queue.enqueueTask(kernel, &waitList[0], &kernelEvent);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		waitList[1].push_back(kernelEvent);

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(dataBuffer, CL_FALSE, 0, sizeof(cl_float) * 4, &data[0], &waitList[1], &readEvent);

		// set event callback functions
		readEvent.setCallback(CL_COMPLETE, &read_complete, (void*)&data[0]);

		printf("Old data: %4.2f, %4.2f, %4.2f, %4.2f\n", data[0], data[1], data[2], data[3]);
		std::cout << "Press ENTER to execute kernel." << std::endl;
		getchar();
		userEvent.setStatus(CL_COMPLETE);

		// wait for enqueued commands to complete
		queue.finish();
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
