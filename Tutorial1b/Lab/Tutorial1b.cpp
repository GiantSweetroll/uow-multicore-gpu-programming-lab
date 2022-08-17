#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

#define NUMBER_OF_DEVICES 1		// minimum number of devices required

// C++ standard library and STL headers
#include <iostream>
#include <vector>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// functions to handle errors
#include "error.h"

// to avoid having to use prefixes
//using namespace std;
//using namespace cl;

int main(void) 
{
	std::vector<cl::Platform> platforms;	// available platforms
	std::vector<cl::Device> devices;		// devices available to a platform
	cl::Context context;					// context for one or more devices
	std::string outputString;				// string for output
	unsigned int i, j;						// counters

	try {
		// get the number of available OpenCL platforms
		cl::Platform::get(&platforms);
		std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl;

		// find the first platform with required number of devices
		for (i = 0; i < platforms.size(); i++) 
		{
			// get all devices available to the platform
			platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

			// check for required number of devices on the platform
			if (devices.size() >= NUMBER_OF_DEVICES)
			{
				std::cout << "--------------------" << std::endl;
				std::cout << "Creating a context for platform " << i <<"..." << std::endl;

				// create a context for all available devices on that platform
				context = cl::Context(devices);

				// check devices in the context
				std::cout << "\nDevices in the context:" << std::endl;

				// get devices in the context
				std::vector<cl::Device> contextDevices = context.getInfo<CL_CONTEXT_DEVICES>();

				// output names of devices in the context
				for (j = 0; j < contextDevices.size(); j++) 
				{
					outputString = contextDevices[j].getInfo<CL_DEVICE_NAME>();
					std::cout << "  Device " << j << ": " << outputString << std::endl;
				}

				// break from loop, since required number of devices found
				break;
			}
		}

		std::cout << "--------------------" << std::endl;

		// if no platform has required number of devices
		if (i == platforms.size()) {
			std::cout << "Unable to find a platform with " << NUMBER_OF_DEVICES << " or more devices." << std::endl;
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