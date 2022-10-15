__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

// 3x3 Vertical Sobel edge detection filter
__constant float vSobelFilter[9] = {-1.0, -2.0, -1.0, 
									 0.0,  0.0,  0.0, 
									 1.0,  2.0,  1.0};
// 3x3 Blurring filter
__constant float BlurringFilter[9] = {	1.0/9, 1.0/9, 1.0/9, 
										1.0/9, 1.0/9, 1.0/9, 
										1.0/9, 1.0/9, 1.0/9};
// 3x3 Sharpening filter
__constant float SharpeningFilter[9] = { 0.0, -1.0, 0.0, 
										-1.0, 5.0, -1.0, 
										 0.0, -1.0, 0.0};

__kernel void simple_conv(read_only image2d_t src_image,
					write_only image2d_t dst_image) {

   // get work-item’s row and column position
   int column = get_global_id(0); 
   int row = get_global_id(1);

   // accumulated pixel value
   float4 sum = (float4)(0.0);

   // filter's current index
   int filter_index =  0;

   int2 coord;
   float4 pixel;

   // iterate over the rows
   for(int i = -1; i <= 1; i++) {
	  coord.y =  row + i;

      // iterate over the columns
	  for(int j = -1; j <= 1; j++) {
         coord.x = column + j;

		 // read value pixel from the image
		 pixel = read_imagef(src_image, sampler, coord);

		 // acculumate weighted sum
		 sum.xyz += pixel.xyz * SharpeningFilter[filter_index++];
	  }
   }

   // write new pixel value to output
   coord = (int2)(column, row); 
   write_imagef(dst_image, coord, sum);
}