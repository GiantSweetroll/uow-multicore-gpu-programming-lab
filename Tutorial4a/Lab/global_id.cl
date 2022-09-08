__kernel void global_id(__global int* output) 
{
   int i = get_global_id(0);			// get work-item's global id
   int offset = get_global_offset(0);	// get offset
   int index = i - offset;				// array index

   printf("Global ID = %d\n", i);		// display global id

   output[index] = i;					// store global id as output
}
