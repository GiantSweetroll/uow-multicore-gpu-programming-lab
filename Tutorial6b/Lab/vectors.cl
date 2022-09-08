__kernel void vectors(__global int* array, 
                      __global int4* vec,
                      __global int* output) {

   int global_id = get_global_id(0);
   int4 temp1, temp2;
   int8 temp3;

   // load vector from array
   temp1 = vload4(global_id, array);

   // get a vector from vec1
   temp2 = vec[global_id];

   // add a value to all vector elements
   temp2 += array[global_id];

   // if all elements in temp2 are greater than elements in temp1
   if(all(temp1 < temp2))
      temp3 = (int8)(temp1, temp2);
   else
      temp3 = (int8)(temp1.s3210, temp2.s3210);

   // store vector in output array
   vstore8(temp3, global_id, output);
}
