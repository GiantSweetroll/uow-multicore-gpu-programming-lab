__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; 

__kernel void gradient(	read_only image2d_t src_image,
						write_only image2d_t dst_image) {

   // get pixel coordinate
   int2 coord = (int2)(get_global_id(0), get_global_id(1));

   // read pixel value
   float4 pixel = read_imagef(src_image, sampler, coord);

   // darken pixel based on coordinate
   pixel.xyz -= (float)coord.y/get_image_height(src_image);

   // write new pixel value to output
   write_imagef(dst_image, coord, pixel);
}
