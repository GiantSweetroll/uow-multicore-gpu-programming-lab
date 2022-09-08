__kernel void memory_spaces(__private int num1,
						    float num2,
						    __local float* shared, 
						    __constant float* input,
						    __global int* output1,
						    __global float* output2) { 

   // private variables for work-item information
   __private int global_id = get_global_id(0);
   int local_id = get_local_id(0);
   int group_id = get_group_id(0);
   int group_size = get_local_size(0);

   // moving from global into local memory
   shared[local_id] = input[global_id];

   // wait for all work-items in the work-group to complete preceeding code
   barrier(CLK_LOCAL_MEM_FENCE);

   // modify data in global memory
   output1[global_id] = num1 + local_id;

   // only do this for the first work-item in a work-group
   if(local_id == 0)
   {
	  // sum the values for the work-group
	  for(int i = 1; i < group_size; i++)
	  {
	     shared[0] += shared[i];
	  }

	  // output some data for the work-group
      output2[group_id] = shared[0] + num2;
   }
}
