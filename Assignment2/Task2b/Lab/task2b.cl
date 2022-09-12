char encrypt(char c, int n)
{
	if (c >= 97 && c <=122)
	{
		c -= 32;
	}

	// only shift alphabetical characters
	if (c >= 65 && c <= 90)
	{
		c += n;

		// If any of the new characters are beyond the alphabet
		// range, loop them back in
		if (c > 90)
		{
			c = 64 + (c % 90);
		}
		else if (c < 65)
		{
			c = 91 - (65 % c);
		}
	}

	return c;
}

__kernel void task2b(__global char* input,
                     int n,
                     __global char* output) 
{
	int i = get_global_id(0);		// get work item global id
	int offset = get_global_offset(0);		// get offset
	int index = i - offset;				// determine array index

	// make alphabets uppercase
	char c = encrypt(input[2 * i], n);
	char c2 = encrypt(input[2 * i + 1], n);
		
	// save to output
	output[2 * index] = c;
	output[2 * index + 1] = c2;
}
