__kernel void task1b(__global int4* input1,
                     __global int* input2,
                     __global int* output) 
{
    int i = get_global_id(0);                // get work item global id

    // Get the data of two consecutive int4 vectors and store them in
    // an int8 vector
    int8 v = (int8)((int4)(input1[2 * i]), (int4)(input1[2 * i+1]));

    // Store values from input2
    int8 v1 = vload8(0, input2);
    int8 v2 = vload8(1, input2);

    int8 results;

    // If any of the elements in v are > 7
    if (any(v > 7))
    {
        // create select mask
        int8 mask;
        mask.s0 = isgreater(v.s0, 7.0) * -1;
        mask.s1 = isgreater(v.s1, 7.0) * -1;
        mask.s2 = isgreater(v.s2, 7.0) * -1;
        mask.s3 = isgreater(v.s3, 7.0) * -1;
        mask.s4 = isgreater(v.s4, 7.0) * -1;
        mask.s5 = isgreater(v.s5, 7.0) * -1;
        mask.s6 = isgreater(v.s6, 7.0) * -1;
        mask.s7 = isgreater(v.s7, 7.0) * -1;

        results = select(v2, v1, mask);
    }
    else
    {
        results = (int8)((int4)(v1.lo), (int4)(v2.lo));
    }

    int offset = get_global_offset(0);      // get offset
    int index = i - offset;                 // determine array index

    // Store value of v, v1, v2, and result
    vstore8(v, index * 4, output);
    vstore8(v1, index * 4, output + 8);
    vstore8(v2, index * 4, output + 16);
    vstore8(results, index * 4, output + 24);
}
