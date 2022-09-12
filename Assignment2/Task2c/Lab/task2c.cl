char encrypt(char c, __global char* lookupMap)
{
	// only change alphabets
	if (c >= 65 && c <= 90 || c >= 97 && c <= 122)
	{
		c = lookupMap[(int)c];
	}

	return c;
}

__kernel void task2c(__global char* input,
                     __global char* lookupMap,
                     __global char* output) 
{
	int i = get_global_id(0);		// get work item global id
	int offset = get_global_offset(0);		// get offset
	int index = i - offset;				// determine array index

	// make alphabets uppercase
	char c = encrypt(input[2 * i], lookupMap);
	char c2 = encrypt(input[2 * i + 1], lookupMap);
		
	// save to output
	output[2 * index] = c;
	output[2 * index + 1] = c2;
}
