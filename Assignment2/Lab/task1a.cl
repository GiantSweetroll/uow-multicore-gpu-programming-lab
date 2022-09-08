__kernel void populate_array(int increment, 
                             __global int* output) {
    int i = get_global_id(0);                // get work item global id
    int offset = get_global_offset(0);      // get offset
    int index = i - offset;                 // determine array index

    int value = increment * index + 3;     // calculate value to be put into the array

    output[index] = value;                  // store the value
}
