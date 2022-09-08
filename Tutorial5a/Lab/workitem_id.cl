__kernel void workitem_id(__global int* input, 
						  __global int* output) { 

   // work-item information
   int global_id = get_global_id(0);
   int offset = get_global_offset(0);
   int global_size = get_global_size(0);

   // work-item and work-group information
   int local_id = get_local_id(0);
   int group_id = get_group_id(0);
   int num_groups = get_num_groups(0);
   int group_size = get_local_size(0);
  
   // check if it is the first work-item
   if(global_id == offset)
   {
      output[0] = global_size;
      output[1] = offset;
      output[2] = num_groups;
      output[3] = group_size;
   }

   // calculate output array index
   int index = (global_id - offset + 1) * 4;

   // each work-item outputs data to its respective output space
   output[index] = global_id;
   output[index+1] = group_id;
   output[index+2] = local_id;
   output[index+3] = input[global_id];
}

