__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

// 7x7 Gaussian Blurring filter
__constant float GaussFilter[49] = {
	0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036,
	0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
	0.001446, 0.014662, 0.058488, 0.092651, 0.092651, 0.014662, 0.001446,
	0.001446, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291,
	0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
	0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
	0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036
};

__kernel void gauss_conv(read_only image2d_t src_image,
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
   for(int i = -3; i <= 3; i++) {
	  coord.y =  row + i;

      // iterate over the columns
	  for(int j = -3; j <= 3; j++) {
         coord.x = column + j;

		 // read value pixel from the image
		 pixel = read_imagef(src_image, sampler, coord);

		 // acculumate weighted sum
		 sum.xyz += pixel.xyz * GaussFilter[filter_index++];
	  }
   }

   // write new pixel value to output
   coord = (int2)(column, row); 
   write_imagef(dst_image, coord, sum);
}