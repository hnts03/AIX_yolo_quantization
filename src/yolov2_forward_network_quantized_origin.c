#include "additionally.h"    // some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

#define GEMMCONV

//#define SSE41
//#undef AVX

#define MAX_VAL_8 (256/2 - 1)    // 7-bit (1-bit sign)
#define MAX_VAL_16 (256*256/2 - 1)    // 15-bit (1-bit sign)
#define MAX_VAL_32 (256*256*256*256/2 - 1) // 31-bit (1-bit sign)

int max_abs(int src, int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}

short int max_abs_short(short int src, short int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}

// im2col.c
int8_t im2col_get_pixel_int8(int8_t *im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

// im2col.c
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_int8(int8_t* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, int8_t* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_int8(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

void gemm_nn_int8_int16(int M, int N, int K, int8_t ALPHA,
    int8_t *A, int lda,
    int8_t *B, int ldb,
    int16_t *C, int ldc)
{
    int32_t *c_tmp = calloc(N, sizeof(int32_t));
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register int16_t A_PART = ALPHA*A[i*lda + k];
            //#pragma simd parallel for
            for (j = 0; j < N; ++j) {
                c_tmp[j] += A_PART*B[k*ldb + j];
            }
        }
        for (j = 0; j < N; ++j) {
            C[i*ldc + j] += max_abs(c_tmp[j], MAX_VAL_16);
            c_tmp[j] = 0;
        }
    }
    free(c_tmp);
}

void gemm_nn_int8_int32(int M, int N, int K, int8_t ALPHA,
    int8_t *A, int lda,
    int8_t *B, int ldb,
    int32_t *C, int ldc)
{
    int32_t *c_tmp = calloc(N, sizeof(int32_t));
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register int16_t A_PART = ALPHA*A[i*lda + k];
            //#pragma simd parallel for
            for (j = 0; j < N; ++j) {
                c_tmp[j] += A_PART*B[k*ldb + j];
            }
        }
        for (j = 0; j < N; ++j) {
            C[i*ldc + j] += max_abs(c_tmp[j], MAX_VAL_32);
            c_tmp[j] = 0;
        }
    }
    free(c_tmp);
}

void forward_convolutional_layer_q(layer l, network_state state)
{

    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, j;
    int const out_size = out_h*out_w;

    typedef int16_t conv_t;    // l.output
    conv_t *output_q = calloc(l.outputs, sizeof(conv_t));

    state.input_int8 = (int8_t *)calloc(l.inputs, sizeof(int));
    int z;
    for (z = 0; z < l.inputs; ++z) {
        int16_t src = state.input[z] * l.input_quant_multiplier;
        state.input_int8[z] = max_abs(src, MAX_VAL_8);
    }

    // Convolution
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    int8_t *a = l.weights_int8;
    int8_t *b = (int8_t *)state.workspace;
    conv_t *c = output_q;    // int16_t

    // Use GEMM (as part of BLAS)
    im2col_cpu_int8(state.input_int8, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
    int t;    // multi-thread gemm
    #pragma omp parallel for
    for (t = 0; t < m; ++t) {
        gemm_nn_int8_int16(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
    }
    free(state.input_int8);

    // Bias addition
    int fil;
    for (fil = 0; fil < l.n; ++fil) {
        for (j = 0; j < out_size; ++j) {
            output_q[fil*out_size + j] = output_q[fil*out_size + j] + l.biases_quant[fil];
        }
    }

    // Activation
    if (l.activation == LEAKY) {
        for (i = 0; i < l.n*out_size; ++i) {
            output_q[i] = (output_q[i] > 0) ? output_q[i] : output_q[i] / 10;
        }
    }

    // De-scaling
    float ALPHA1 = 1 / (l.input_quant_multiplier * l.weights_quant_multiplier);
    for (i = 0; i < l.outputs; ++i) {
        l.output[i] = output_q[i] * ALPHA1;
    }

    // saving l.output == next layer input
    // int abc;
    // char outfile[30];
    // sprintf(outfile, "weights/CONV%d_OUT.txt",l.index);
    // FILE *fp_out = fopen(outfile, "w");
    // for (abc = 0; abc < out_size *  l.n; abc++){
    //     fprintf(fp_out, "%x", l.output[abc]);
    // }


    free(output_q);
}

void yolov2_forward_network_q(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i) {
        state.index = i;
        layer l = net.layers[i];

        if (l.type == CONVOLUTIONAL) {
            forward_convolutional_layer_q(l, state);
        }
        else if (l.type == MAXPOOL) {
            forward_maxpool_layer_cpu(l, state);
        }
        else if (l.type == ROUTE) {
            forward_route_layer_cpu(l, state);
        }
        else if (l.type == REORG) {
            forward_reorg_layer_cpu(l, state);
        }
        else if (l.type == UPSAMPLE) {
            forward_upsample_layer_cpu(l, state);
        }
        else if (l.type == SHORTCUT) {
            forward_shortcut_layer_cpu(l, state);
        }
        else if (l.type == YOLO) {
            forward_yolo_layer_cpu(l, state);
        }
        else if (l.type == REGION) {
            forward_region_layer_cpu(l, state);
        }
        else {
            printf("\n layer: %d \n", l.type);
        }
        state.input = l.output;

        // Saving output
        // int out_h = (l.h + 2 * l.pad - l.size) + 1;
        // int out_w = (l.w + 2 * l.pad - l.size) + 1;
        // int output_size = out_h * out_w;

        // int k;
        // char outfile[30];
        // sprintf(outfile, "weights/CONV%d_OUT.txt",i);
        // FILE *fp_out = fopen(outfile, "w");
        // for (k = 0; k < output_size; k++){
        //     fprintf(fp_out, "%x\n", l.output[k]);
        // }

    }
}

// detect on CPU
float *network_predict_quantized(network net, float *input)
{
    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    yolov2_forward_network_q(net, state);    // network on CPU
                                            //float *out = get_network_output(net);
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}

//mean function for z-norm
float _mean(float *weights, int filter_num, int weight_size){
    float sum = 0;
    float mean;
    for (int i = 0; i < weight_size; i++){
        sum += weights[filter_num * weight_size + i];
    }
    mean = sum / weight_size;
    return mean;
}

// variance function for z-norm
float _variance(float *weights, int filter_num, float mean, int weight_size){
    float deviation_square_sum = 0;
    float variance;
    for (int i=0; i < weight_size; i++){
        deviation_square_sum += pow(weights[filter_num * weight_size + i] - mean, 2);
    }
    variance = deviation_square_sum / weight_size;
    return variance;
}

// standard deviation function for z-norm
float _std_deviation(float variance){
    float std_deviation = sqrt(variance);
    return std_deviation;
}

// get min-max from weights
float get_minmax(layer *l, int weights_size){
    float max = 0;
    float min = 0;
    for (int i = 0; i < weights_size; i++){
        if (max < l->weights[i]) max = l->weights[i];
        if (min > l->weights[i]) min = l->weights[i];
    }
    printf("The value of minmax : %f\n", max - min);
    return max - min;
}

// get mean
float get_mean(layer *l, int weights_size){
    float sum = 0;
    for (int i=0; i<weights_size; i++){
        sum += l->weights[i];
    }

    return sum / weights_size;
}

//get variance
float get_variance(layer *l, int weights_size){
    float deviation_square_sum = 0;
    float variance;
    float mean = get_mean(l, weights_size);
    for (int i=0; i < weights_size; i++){
        deviation_square_sum += pow(l->weights[i] - mean, 2);
    }
    variance = deviation_square_sum / weights_size;
    return variance;
}

// get std_deviation
float get_std_deviation(layer *l, int weights_size){
    return sqrt(get_variance(l, weights_size));
}

// This function contaminates original weights.
// Have to additional work for contamination problem.
void do_normalization(layer *l, char *method){
    // get copy of  weights
    printf("check 2\n");
    
    size_t const weights_size = l->size*l->size*l->c*l->n;
    size_t const filter_size = l->size*l->size*l->c;
    float copied_weights[weights_size];
    
    int fil, i;
    // memcpy(copied_weights, l->weights, sizeof(l->weights));
    printf("check 3\n");
    for (fil = 0; fil < l->n; ++fil) {
        for (i = 0; i < filter_size; ++i) {
            copied_weights[fil*filter_size + i] = l->weights[fil*filter_size + i];
        }
    }
    // debug handling part
    // printf("*** This is preview for weights ***\n");
    // for (i=0; i<10; i++){
    //     printf("weight[%d] : [%f]\n", i, copied_weights[i]);
    // }

    // if method is minmax
    // 22.03.04 below minmax working distinguish of filter.
    if (strcmp(method, "minmax") == 0){
        
        float norm_weight;
        int j;

        
        
        // per filter
        for (fil = 0; fil < l->n; ++fil) {
            float max = 0;
            float min = 0;
            // get min, max of weights
            for (i = 0; i < filter_size; ++i) {
                if (max < copied_weights[fil*filter_size + i]) max = copied_weights[fil*filter_size + i];
                if (min > copied_weights[fil*filter_size + i]) min = copied_weights[fil*filter_size + i];
            }
        
            printf("min, max : %f %f\n", min, max);

            // do min-max normalization
            for (i = 0; i < filter_size; ++i) {
                norm_weight = (copied_weights[fil*filter_size + i] - min) / (max - min);
                copied_weights[fil*filter_size + i] = norm_weight;
            }
            
        }

        // debug handling part
        // printf("*** This is preview for normalized weights ***\n");
        // for (i=0; i<10; i++){
        //     printf("weight[%d] : [%f]\n", i, copied_weights[i]);
        // }

        // re-copy to origin weight
        // memcpy(l->weights, copied_weights, sizeof(l->weights));
        for (fil = 0; fil < l->n; ++fil) {
            for (i = 0; i < filter_size; ++i) {
                l->weights[fil*filter_size + i] = copied_weights[fil*filter_size + i];
            }
        }
    }
    
    // if method is znorm
    // 22.03.05 below code is about znorm 
    if (strcmp(method, "znorm") == 0){

        // per filter
        for (fil = 0; fil < l->n; ++fil) {
            // get mean, variance, std_deviation
            float mean = _mean(copied_weights, fil, filter_size);
            float variance = _variance(copied_weights, fil, mean, filter_size);
            float std_deviation = _std_deviation(variance);            
        
            printf("[filter num : %d][mean : %f] [variance : %f] [standard deviation : %f]\n", fil, mean, variance, std_deviation);

            // do z-normalization
            for (i = 0; i < filter_size; ++i) {
                float norm_weight = (copied_weights[fil*filter_size + i] - mean) / std_deviation;
                copied_weights[fil*filter_size + i] = norm_weight;
            }
            
        }

        // re-copy to origin weight
        for (i = 0; i < weights_size; i++){
            l->weights[i] = copied_weights[i];
        }
    }
}

/* Quantization-related */

void do_quantization(network net) {
    int counter = 0;
    char* method = "znorm"; // minmax, znorm

    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        /*
        TODO: implement quantization 
        The implementation given below is a naive version of per-network quantization; implement your own quantization that minimizes the mAP degradation
        */

        printf("\n");
        
        if (l->type == CONVOLUTIONAL) { // Quantize conv layer only
            size_t const weights_size = l->size*l->size*l->c*l->n;
            size_t const filter_size = l->size*l->size*l->c;
            // float layer_std_weight[11] = {};
            
            float layer_minmax_weight[11] = {256.032, 92.29997, 92.29997, 92.29997, 92.29997, 92.29997, 92.29997, 92.29997, 92.29997, 92.29997, 92.29997};
            // {1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1} => 64.16%
            // {1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1} => 64.06%
            // {1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1} => 64.19%
            // {1, 2, 4, 1, 1, 4, 1, 1, 1, 1, 1} => 64.22%
            // {1, 2, 2, 2, 1, 4, 1, 1, 1, 1, 1} => 64.28%
            // {1, 2, 2, 1, 1, 4, 1, 1, 1, 1, 1} => 64.52%
            // {1, 2, 2, 1, 1, 4, 4, 1, 1, 1, 1} => 65.28%
            // {1, 2, 2, 1, 1, 4, 2, 1, 1, 1, 1} => 65.92%
            // {1, 2, 2, 1, 1, 4, 3, 1, 1, 1, 1} => 66.54%
            // {1, 2, 2, 1, 1, 4, 3, 1, 1.5, 1, 1} => 66.75%
            // {1, 2, 2, 1, 1, 4, 3, 1, 1.5, 2, 1} => 67.23%
            
            // {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1} => 64%
            // {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1} => 65%
            // {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1} => 69.74%
            // {4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1} => 70.35%
            // {4.0005, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1} => 70.63%

            // below codes are for debugging
            // printf("%d\n", sizeof(l->weights)*weights_size);
            // printf("%d\n", sizeof(l->weights));
            // printf("%d\n", sizeof(l->weights[0]));
            // printf("%d\n", sizeof(float));

            int i, fil;

            // Input Scaling
            if (counter >= net.input_calibration_size) {
                printf(" Warning: CONV%d has no corresponding input_calibration parameter - default value 16 will be used;\n", j);
            }
            l->input_quant_multiplier = (counter < net.input_calibration_size) ? net.input_calibration[counter] : 16;  // Using 16 as input_calibration as default value
            // l->input_quant_multiplier = floor(l->input_quant_multiplier*pow(2,12))/pow(2,12);
            ++counter;

            printf("check 1\n");
            // Weight Quantization
            // do_normalization(l, method);
            // for (fil = 0; fil < l->n; ++fil) {
            //     for (i = 0; i < filter_size; ++i) {
            //         float w = (l->weights[fil*filter_size + i] - 0.5) * 127; // Scale
            //         l->weights_int8[fil*filter_size + i] = max_abs(w, MAX_VAL_8); // Clip
            //     }
            // }


            // Below annotation is a pure-skeleton code.
            
            l->weights_quant_multiplier = 1 / get_minmax(l, weights_size) * layer_minmax_weight[counter - 1];
            // l->weights_quant_multiplier = 1 / get_std_deviation(l, weights_size); // Arbitrarily set to 32; you should devise your own method to calculate the weight multiplier
            for (fil = 0; fil < l->n; ++fil) {
                for (i = 0; i < filter_size; ++i) {
                    float w = l->weights[fil*filter_size + i] * l->weights_quant_multiplier; // Scale
                    l->weights_int8[fil*filter_size + i] = max_abs(w, MAX_VAL_8); // Clip
                }
            }
            

            // Bias Quantization
            float biases_multiplier = (l->weights_quant_multiplier * l->input_quant_multiplier);
            for (fil = 0; fil < l->n; ++fil) {
                float b = l->biases[fil] * biases_multiplier; // Scale
                l->biases_quant[fil] = max_abs(b, MAX_VAL_16); // Clip
            }

            printf(" CONV%d multipliers: input %g, weights %g, bias %g \n", j, l->input_quant_multiplier, l->weights_quant_multiplier, biases_multiplier);
        }
        else {
            printf(" No quantization for layer %d (layer type: %d) \n", j, l->type);
        }
    }
}

// Save quantized weights, bias, and scale
void save_quantized_model(network net) {
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];
        if (l->type == CONVOLUTIONAL) {
            size_t const weights_size = l->size*l->size*l->c*l->n;
            size_t const filter_size = l->size*l->size*l->c;

            printf(" Saving quantized weights, bias, and scale for CONV%d \n", j);

            char weightfile[30];
            char biasfile[30];
            char scalefile[30];
            char origin_weightfile[30];
            char outfile[30];

            sprintf(weightfile, "weights/CONV%d_W.txt", j);
            sprintf(biasfile, "weights/CONV%d_B.txt", j);
            sprintf(scalefile, "weights/CONV%d_S.txt", j);
            sprintf(origin_weightfile, "weights/CONV%d_ORIGIN.txt", j);
            // sprintf(outfile, "weights/CONV%d_OUT.txt", j);

            int k;

            // int out_h = (l->h + 2 * l->pad - l->size) / l->stride + 1;
            // int out_w = (l->w + 2 * l->pad - l->size) / l->stride + 1;
            // int output_size = out_h * out_w;
            
            // FILE *fp_out = fopen(outfile, "w");
            // for (k = 0; k < output_size * l->n; k++){
            //     fprintf(fp_out, "%x\n", l->output[k]);
            // }
            
            

            FILE *fp_ori = fopen(origin_weightfile, "w");
            for (k = 0; k < weights_size; k++){
                // float origin_weight = k < weight_size ? l->weights[k] : 0;
                fprintf(fp_ori, "%f\n", l->weights[k]);
            }
            fclose(fp_ori);
            

            FILE *fp_w = fopen(weightfile, "w");
            for (k = 0; k < weights_size; k = k + 4) {
                uint8_t first = k < weights_size ? l->weights_int8[k] : 0;
                uint8_t second = k+1 < weights_size ? l->weights_int8[k+1] : 0;
                uint8_t third = k+2 < weights_size ? l->weights_int8[k+2] : 0;
                uint8_t fourth = k+3 < weights_size ? l->weights_int8[k+3] : 0;
                fprintf(fp_w, "%02x%02x%02x%02x\n", first, second, third, fourth);
            }
            fclose(fp_w);

            FILE *fp_b = fopen(biasfile, "w");
            for (k = 0; k < l->n; k = k + 4) {
                uint16_t first = k < l->n ? l->biases_quant[k] : 0;
                uint16_t second = k+1 < l->n ? l->biases_quant[k+1] : 0;
                fprintf(fp_b, "%04x%04x\n", first, second);
            }
            fclose(fp_b);

            FILE *fp_s = fopen(scalefile, "w");
            fprintf(fp_s, "%f\n", l->input_quant_multiplier);
            fclose(fp_s);
        }
    }
}