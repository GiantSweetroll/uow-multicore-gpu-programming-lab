__kernel void op_test(__global int4* output) {

   // initialising a vector
   int4 vec = (int4)(1, 2, 3, 4);
   
   // adding 4 to every element of the vector
   vec += 4;
   
   // sets the third element of the vector to 0
   // -1 in hexadecimal = 0xFFFFFFFF
   if(vec.s2 == 7)
      vec &= (int4)(-1, -1, 0, -1);
   
   // sets the first element to -1 and the second element to 0
   vec.s01 = vec.s23 < 7; 
   
   // divides the last element by 2 until it is less than or equal to 7
   while(vec.s3 > 7 && (vec.s0 < 16 || vec.s1 < 16))
      vec.s3 >>= 1; 
      
   output[0] = vec;
}
