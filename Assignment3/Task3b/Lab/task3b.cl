__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__constant float Weights[7] = {
	0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598
};

__kernel void task3b(read_only image2d_t src_image,
						write_only image2d_t dst_image,
						int pass_type) {
	
	// get work-item's row and column positoon
	int column = get_global_id(0);
	int row = get_global_id(1);

	// accumulated pixel value
	float4 sum = (float4)(0.0);

	// filter's current index
	int filter_index = 0;

	int2 coord;
	float4 pixel;

	// Horizontal pass
	if (pass_type == 0) {
		coord.x = column;
	} 
	// Vertical pass
	else {
		coord.y = row;
	}

	// iterate over the pixels
	for (int i = -3; i <= 3; i++) {

		// perform horizontal pass
		if (pass_type == 1) {
			coord.y = row + i;
		} 
		// perform vertical pass
		else {
			coord.x = column + i;
		}

		// read pixel value
		pixel = read_imagef(src_image, sampler, coord);

		// accumulate weighted sum
		sum.xyz += pixel.xyz * Weights[filter_index++];
	}

	// write new pixel value to output
	coord = (int2) (column, row);
	write_imagef(dst_image, coord, sum);
}