#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

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
	std::string outputString;				// string for output
	unsigned int i, j;						// counters

	try {
		// get the number of available OpenCL platforms
		cl::Platform::get(&platforms);
		std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl;

		// for each platform
		for (i = 0; i < platforms.size(); i++) 
		{
			std::cout << "--------------------" << std::endl;
			// output platform index
			std::cout << "  Platform " << i << ":" << std::endl;

			// get and output platform name
			platforms[i].getInfo(CL_PLATFORM_NAME, &outputString);
			std::cout << "\tName: " << outputString << std::endl;

			// get and output platform vendor name
			outputString = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
			std::cout << "\tVendor: " << outputString << std::endl;

			// get and output OpenCL version supported by the platform
			outputString = platforms[i].getInfo<CL_PLATFORM_VERSION>();
			std::cout << "\tVersion: " << outputString << std::endl;

			// get all devices available to the platform
			platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

			std::cout << "\nNumber of devices available to platform " << i << ": " << devices.size() << std::endl;
			std::cout << "--------------------" << std::endl;

			// for each device
			for (j = 0; j < devices.size(); j++) 
			{
				// output device index
				std::cout << "  Device " << j << std::endl;

				// get and output device name
				outputString = devices[j].getInfo<CL_DEVICE_NAME>();
				std::cout << "\tName: " << outputString << std::endl;

				// get and output device type
				cl_device_type type;
				devices[j].getInfo(CL_DEVICE_TYPE, &type);
				if (type == CL_DEVICE_TYPE_CPU)
					std::cout << "\tType: " << "CPU" << std::endl;
				else if (type == CL_DEVICE_TYPE_GPU)
					std::cout << "\tType: " << "GPU" << std::endl;
				else
					std::cout << "\tType: " << "Other" << std::endl;

				// get and output device vendor
				outputString = devices[j].getInfo<CL_DEVICE_VENDOR>();
				std::cout << "\tVendor: " << outputString << std::endl;

				// get and output OpenCL version supported by the device
				outputString = devices[j].getInfo<CL_DEVICE_VERSION>();
				std::cout << "\tVersion: " << outputString << std::endl;
			}
			std::cout << "--------------------" << std::endl;
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