float8 shuffle_test(uint8 mask, float4 input)
{
    return shuffle(input, mask);
}

char16 shuffle2_test()
{
   // initialise mask and input vectors
   uchar16 mask = (uchar16)(6, 10, 5, 2, 8, 0, 9, 14, 7, 5, 12, 3, 11, 15, 1, 13);
   char8 input1 = (char8)('l', 'o', 'f', 'c', 'a', 'u', 's', 'f');
   char8 input2 = (char8)('f', 'e', 'h', 't', 'n', 'n', '2', 'i');

   return shuffle2(input1, input2, mask);
}

float4 select_test()
{
   // initialise mask and input vectors
   int4 mask = (int4)(-1, 0, -1, 0);
   float4 input1 = (float4)(0.25f, 0.5f, 0.75f, 1.0f);
   float4 input2 = (float4)(1.25f, 1.5f, 1.75f, 2.0f);

   return select(input1, input2, mask);
}

uchar2 bitselect_test()
{
   // initialise mask and input vectors
   uchar2 mask = (uchar2)(0xAA, 0x55);
   uchar2 input1 = (uchar2)(0x0F, 0x0F);
   uchar2 input2 = (uchar2)(0x33, 0x33);

   return bitselect(input1, input2, mask); 
}

__kernel void shuffle_select(__global float8* s1, 
                             __global char16* s2,
                             __global float4* s3, 
                             __global uchar2* s4) {

   // shuffle example
   uint8 mask = (uint8)(0, 1, 2, 3, 1, 3, 0, 2);
   float4 input = (float4)(0.25f, 0.5f, 0.75f, 1.0f);
   *s1 = shuffle_test(mask, input); 
   
   // shuffle2 example
   *s2 = shuffle2_test();

   // select example
   *s3 = select_test();

   // bitselect example
   *s4 = bitselect_test(); 
}
