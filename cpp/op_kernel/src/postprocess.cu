#include "postprocess.h"

#include <cstdio>

// transpose kernel - 将检测输出从 [numBboxes, numElements] 转置为 [numElements, numBboxes]
__global__ void transpose_kernel(float * src,
                                 float * dst,
                                 int     numBboxes,
                                 int     numElements,
                                 int     edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) {
        return;
    }

    dst[position] = src[(position % numElements) * numBboxes + position / numElements];
}

void transpose(float * src, float * dst, int numBboxes, int numElements, cudaStream_t stream) {
    int edge      = numBboxes * numElements;
    int blockSize = 256;
    int gridSize  = (edge + blockSize - 1) / blockSize;
    transpose_kernel<<<gridSize, blockSize, 0, stream>>>(src, dst, numBboxes, numElements, edge);
}

// decode kernel - 解析检测框坐标、类别置信度和标签，过滤低置信度结果
__global__ void decode_kernel(float * src,
                              float * dst,
                              int     numBboxes,
                              int     numClasses,
                              float   confThresh,
                              int     maxObjects,
                              int     numBoxElement) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= numBboxes) {
        return;
    }

    float * pitem      = src + (4 + numClasses) * position;
    float * classConf  = pitem + 4;
    float   confidence = 0;
    int     label      = 0;
    for (int i = 0; i < numClasses; i++) {
        if (classConf[i] > confidence) {
            confidence = classConf[i];
            label      = i;
        }
    }

    if (confidence < confThresh) {
        return;
    }

    // atomicAdd 保证多线程并发写入时不冲突：返回写入前的旧值作为当前框的写入位置
    int index = (int) atomicAdd(dst, 1);
    if (index >= maxObjects) {
        return;
    }

    float cx     = pitem[0];
    float cy     = pitem[1];
    float width  = pitem[2];
    float height = pitem[3];

    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    float * pout_item = dst + 1 + index * numBoxElement;
    pout_item[0]      = left;
    pout_item[1]      = top;
    pout_item[2]      = right;
    pout_item[3]      = bottom;
    pout_item[4]      = confidence;
    pout_item[5]      = label;
    pout_item[6]      = 1;  // 1 = keep, 0 = ignore
}

void decode(float *      src,
            float *      dst,
            int          numBboxes,
            int          numClasses,
            float        confThresh,
            int          maxObjects,
            int          numBoxElement,
            cudaStream_t stream) {
    cudaMemsetAsync(dst, 0, sizeof(int), stream);
    int blockSize = 256;
    int gridSize  = (numBboxes + blockSize - 1) / blockSize;
    decode_kernel<<<gridSize, blockSize, 0, stream>>>(src, dst, numBboxes, numClasses, confThresh,
                                                      maxObjects, numBoxElement);
}

// NMS - 交并比计算，用于非极大值抑制的框重叠度评估
__device__ float box_iou(float aleft,
                         float atop,
                         float aright,
                         float abottom,
                         float bleft,
                         float btop,
                         float bright,
                         float bbottom) {
    float cleft   = max(aleft, bleft);
    float ctop    = max(atop, btop);
    float cright  = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) {
        return 0.0f;
    }

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

__global__ void nms_kernel(float * data, float kNmsThresh, int maxObjects, int numBoxElement) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    int count    = min((int) data[0], maxObjects);
    if (position >= count) {
        return;
    }

    // left, top, right, bottom, confidence, class, keepflag
    float * pcurrent = data + 1 + position * numBoxElement;
    float * pitem;
    for (int i = 0; i < count; i++) {
        pitem = data + 1 + i * numBoxElement;
        if (i == position || pcurrent[5] != pitem[5]) {
            continue;
        }

        // NMS 抑制规则：同一类别中，置信度更高的框抑制低置信度框
        // 置信度相同时按线程 ID 打破平局，避免双方互相抑制
        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position) {
                continue;
            }

            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0],
                                pitem[1], pitem[2], pitem[3]);

            if (iou > kNmsThresh) {
                pcurrent[6] = 0;  // 1 = keep, 0 = ignore
                return;
            }
        }
    }
}

__constant__ uchar3 C_DEVICE_COLOR_MAP[256];  // BGR

__global__ void normlize_color_kernel(float *  src,
                                      uchar *  dst,
                                      uchar3 * dst_colormap,
                                      int      input_w,
                                      int      input_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= input_w || idy >= input_h) {
        return;
    }
    // normalize
    // 防止Nan
    float norm_val = src[idy * input_w + idx];
    uchar gray_val = (uchar) (norm_val * 255.0f + 0.5f);

    dst[idy * input_w + idx] = gray_val;

    // colormap
    dst_colormap[idy * input_w + idx] = C_DEVICE_COLOR_MAP[gray_val];  // BGR
}

__global__ void resize_kernel(uchar *  src_depth,
                              uchar3 * src_colormap,
                              uchar *  dst_depth,
                              uchar3 * dst_colormap,
                              int      input_w,
                              int      input_h,
                              int      resized_w,
                              int      resized_h,
                              float    resize_scale_w,
                              float    resize_scale_h) {
    int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_idx >= resized_w || dst_idy >= resized_h) {
        return;
    }

    float resize_src_x;
    float resize_src_y;
    int   src_idx;
    int   src_idy;

    // 中心对齐双线性插值：将源坐标 +0.5 偏移后再缩放，避免边缘像素偏移
    resize_src_x = (dst_idx + 0.5f) * resize_scale_w - 0.5f;
    resize_src_y = (dst_idy + 0.5f) * resize_scale_h - 0.5f;

    src_idx = (int) floorf(resize_src_x);
    src_idy = (int) floorf(resize_src_y);

    resize_src_x = resize_src_x - src_idx;
    resize_src_y = resize_src_y - src_idy;
    float fx1y1  = resize_src_x * resize_src_y;
    float fx0y0  = 1.0f - resize_src_x - resize_src_y + fx1y1;
    float fx1y0  = resize_src_x - fx1y1;
    float fx0y1  = resize_src_y - fx1y1;

    dst_depth[dst_idy * resized_w + dst_idx] =
        src_depth[src_idy * input_w + src_idx] * fx0y0 +
        src_depth[src_idy * input_w + src_idx + 1] * fx1y0 +
        src_depth[(src_idy + 1) * input_w + src_idx] * fx0y1 +
        src_depth[(src_idy + 1) * input_w + src_idx + 1] * fx1y1;

    // 计算目标像素的索引
    int dst_continue_idx = dst_idy * resized_w + dst_idx;
    int src_continue_idx = src_idy * input_w + src_idx;

    // 对 X 通道 (B) 进行双线性插值
    dst_colormap[dst_continue_idx].x = src_colormap[src_continue_idx].x * fx0y0 +
                                       src_colormap[src_continue_idx + 1].x * fx1y0 +
                                       src_colormap[src_continue_idx + input_w].x * fx0y1 +
                                       src_colormap[src_continue_idx + input_w + 1].x * fx1y1;

    // 对 Y 通道 (G) 进行双线性插值
    dst_colormap[dst_continue_idx].y = src_colormap[src_continue_idx].y * fx0y0 +
                                       src_colormap[src_continue_idx + 1].y * fx1y0 +
                                       src_colormap[src_continue_idx + input_w].y * fx0y1 +
                                       src_colormap[src_continue_idx + input_w + 1].y * fx1y1;

    // 对 Z 通道 (R) 进行双线性插值
    dst_colormap[dst_continue_idx].z = src_colormap[src_continue_idx].z * fx0y0 +
                                       src_colormap[src_continue_idx + 1].z * fx1y0 +
                                       src_colormap[src_continue_idx + input_w].z * fx0y1 +
                                       src_colormap[src_continue_idx + input_w + 1].z * fx1y1;
}

void nms(float * data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream) {
    int blockSize = maxObjects < 256 ? maxObjects : 256;
    int gridSize  = (maxObjects + blockSize - 1) / blockSize;
    nms_kernel<<<gridSize, blockSize, 0, stream>>>(data, kNmsThresh, maxObjects, numBoxElement);
}

void normalize_colormap_resize(float *      src,
                               uchar *      norm_depth,
                               uchar3 *     norm_colormap,
                               uchar *      dst_depth,
                               uchar3 *     dst_colormap,
                               int          input_w,
                               int          input_h,
                               int          resized_w,
                               int          resized_h,
                               cudaStream_t stream) {
    dim3 block_size(32, 8);
    dim3 grid_size((input_w + 31) >> 5, (input_h + 7) >> 3);

    // 深度归一化 + 颜色映射 + resize：三个步骤在同一 stream 上顺序执行，无同步点
    normlize_color_kernel<<<grid_size, block_size, 0, stream>>>(src, norm_depth, norm_colormap,
                                                                input_w, input_h);
    grid_size = dim3((resized_w + 31) >> 5, (resized_h + 7) >> 3);
    resize_kernel<<<grid_size, block_size, 0, stream>>>(
        norm_depth, norm_colormap, dst_depth, dst_colormap, input_w, input_h, resized_w, resized_h,
        (float) input_w / resized_w, (float) input_h / resized_h);
}

void initColorMapTable() {
    // INFERNO 颜色映射表：将 [0,255] 灰度深度值映射为伪彩色（深紫黑→暗红→亮橙→亮黄）
    // 数据从 matplotlib 提取，编译为 device __constant__ 内存供所有 kernel 只读访问
    static const float r[] = {
        0.001462f, 0.002267f, 0.003299f, 0.004547f, 0.006006f, 0.007676f, 0.009561f, 0.011663f,
        0.013995f, 0.016561f, 0.019373f, 0.022447f, 0.025793f, 0.029432f, 0.033385f, 0.037668f,
        0.042253f, 0.046915f, 0.051644f, 0.056449f, 0.061340f, 0.066331f, 0.071429f, 0.076637f,
        0.081962f, 0.087411f, 0.092990f, 0.098702f, 0.104551f, 0.110536f, 0.116656f, 0.122908f,
        0.129285f, 0.135778f, 0.142378f, 0.149073f, 0.155850f, 0.162689f, 0.169575f, 0.176493f,
        0.183429f, 0.190367f, 0.197297f, 0.204209f, 0.211095f, 0.217949f, 0.224763f, 0.231538f,
        0.238273f, 0.244967f, 0.251620f, 0.258234f, 0.264810f, 0.271347f, 0.277850f, 0.284321f,
        0.290763f, 0.297178f, 0.303568f, 0.309935f, 0.316282f, 0.322610f, 0.328921f, 0.335217f,
        0.341500f, 0.347771f, 0.354032f, 0.360284f, 0.366529f, 0.372768f, 0.379001f, 0.385228f,
        0.391453f, 0.397674f, 0.403894f, 0.410113f, 0.416331f, 0.422549f, 0.428768f, 0.434987f,
        0.441207f, 0.447428f, 0.453651f, 0.459875f, 0.466100f, 0.472328f, 0.478558f, 0.484789f,
        0.491022f, 0.497257f, 0.503493f, 0.509730f, 0.515967f, 0.522206f, 0.528444f, 0.534683f,
        0.540920f, 0.547157f, 0.553392f, 0.559624f, 0.565854f, 0.572081f, 0.578304f, 0.584521f,
        0.590734f, 0.596940f, 0.603139f, 0.609330f, 0.615513f, 0.621685f, 0.627847f, 0.633998f,
        0.640135f, 0.646260f, 0.652369f, 0.658463f, 0.664540f, 0.670599f, 0.676638f, 0.682656f,
        0.688653f, 0.694627f, 0.700576f, 0.706500f, 0.712396f, 0.718264f, 0.724103f, 0.729909f,
        0.735683f, 0.741423f, 0.747127f, 0.752794f, 0.758422f, 0.764010f, 0.769556f, 0.775059f,
        0.780517f, 0.785929f, 0.791293f, 0.796607f, 0.801871f, 0.807082f, 0.812239f, 0.817341f,
        0.822386f, 0.827372f, 0.832299f, 0.837165f, 0.841969f, 0.846709f, 0.851384f, 0.855992f,
        0.860533f, 0.865006f, 0.869409f, 0.873741f, 0.878001f, 0.882188f, 0.886302f, 0.890341f,
        0.894305f, 0.898192f, 0.902003f, 0.905735f, 0.909390f, 0.912966f, 0.916462f, 0.919879f,
        0.923215f, 0.926470f, 0.929644f, 0.932737f, 0.935747f, 0.938675f, 0.941521f, 0.944285f,
        0.946965f, 0.949562f, 0.952075f, 0.954506f, 0.956852f, 0.959114f, 0.961293f, 0.963387f,
        0.965397f, 0.967322f, 0.969163f, 0.970919f, 0.972590f, 0.974176f, 0.975677f, 0.977092f,
        0.978422f, 0.979666f, 0.980824f, 0.981895f, 0.982881f, 0.983779f, 0.984591f, 0.985315f,
        0.985952f, 0.986502f, 0.986964f, 0.987337f, 0.987622f, 0.987819f, 0.987926f, 0.987945f,
        0.987874f, 0.987714f, 0.987464f, 0.987124f, 0.986694f, 0.986175f, 0.985566f, 0.984865f,
        0.984075f, 0.983196f, 0.982228f, 0.981173f, 0.980032f, 0.978806f, 0.977497f, 0.976108f,
        0.974638f, 0.973088f, 0.971468f, 0.969783f, 0.968041f, 0.966243f, 0.964394f, 0.962517f,
        0.960626f, 0.958720f, 0.956834f, 0.954997f, 0.953215f, 0.951546f, 0.950018f, 0.948683f,
        0.947594f, 0.946809f, 0.946392f, 0.946403f, 0.946903f, 0.947937f, 0.949545f, 0.951740f,
        0.954529f, 0.957896f, 0.961812f, 0.966249f, 0.971162f, 0.976511f, 0.982257f, 0.988362f
    };
    static const float g[] = {
        0.000466f, 0.001270f, 0.002249f, 0.003392f, 0.004692f, 0.006136f, 0.007713f, 0.009417f,
        0.011225f, 0.013136f, 0.015133f, 0.017199f, 0.019331f, 0.021503f, 0.023702f, 0.025921f,
        0.028139f, 0.030324f, 0.032474f, 0.034569f, 0.036590f, 0.038504f, 0.040294f, 0.041905f,
        0.043328f, 0.044556f, 0.045583f, 0.046402f, 0.047008f, 0.047399f, 0.047574f, 0.047536f,
        0.047293f, 0.046856f, 0.046242f, 0.045468f, 0.044559f, 0.043554f, 0.042489f, 0.041402f,
        0.040329f, 0.039309f, 0.038400f, 0.037632f, 0.037030f, 0.036615f, 0.036405f, 0.036405f,
        0.036621f, 0.037055f, 0.037705f, 0.038571f, 0.039647f, 0.040922f, 0.042353f, 0.043933f,
        0.045644f, 0.047470f, 0.049396f, 0.051407f, 0.053490f, 0.055634f, 0.057827f, 0.060060f,
        0.062325f, 0.064616f, 0.066925f, 0.069247f, 0.071579f, 0.073915f, 0.076253f, 0.078591f,
        0.080927f, 0.083257f, 0.085580f, 0.087896f, 0.090203f, 0.092501f, 0.094790f, 0.097069f,
        0.099338f, 0.101597f, 0.103848f, 0.106089f, 0.108322f, 0.110547f, 0.112764f, 0.114974f,
        0.117179f, 0.119379f, 0.121575f, 0.123769f, 0.125960f, 0.128150f, 0.130341f, 0.132534f,
        0.134729f, 0.136929f, 0.139134f, 0.141346f, 0.143567f, 0.145797f, 0.148039f, 0.150294f,
        0.152563f, 0.154848f, 0.157151f, 0.159474f, 0.161817f, 0.164184f, 0.166575f, 0.168992f,
        0.171438f, 0.173914f, 0.176421f, 0.178962f, 0.181539f, 0.184153f, 0.186807f, 0.189501f,
        0.192239f, 0.195021f, 0.197851f, 0.200728f, 0.203656f, 0.206636f, 0.209670f, 0.212759f,
        0.215906f, 0.219112f, 0.222378f, 0.225706f, 0.229097f, 0.232554f, 0.236077f, 0.239667f,
        0.243327f, 0.247056f, 0.250856f, 0.254728f, 0.258674f, 0.262692f, 0.266786f, 0.270954f,
        0.275197f, 0.279517f, 0.283913f, 0.288385f, 0.292933f, 0.297559f, 0.302260f, 0.307038f,
        0.311892f, 0.316822f, 0.321827f, 0.326906f, 0.332060f, 0.337287f, 0.342586f, 0.347957f,
        0.353399f, 0.358911f, 0.364492f, 0.370140f, 0.375856f, 0.381636f, 0.387481f, 0.393389f,
        0.399359f, 0.405389f, 0.411479f, 0.417627f, 0.423831f, 0.430091f, 0.436405f, 0.442772f,
        0.449191f, 0.455660f, 0.462178f, 0.468744f, 0.475356f, 0.482014f, 0.488716f, 0.495462f,
        0.502249f, 0.509078f, 0.515946f, 0.522853f, 0.529798f, 0.536780f, 0.543798f, 0.550850f,
        0.557937f, 0.565057f, 0.572209f, 0.579392f, 0.586606f, 0.593849f, 0.601122f, 0.608422f,
        0.615750f, 0.623105f, 0.630485f, 0.637890f, 0.645320f, 0.652773f, 0.660250f, 0.667748f,
        0.675267f, 0.682807f, 0.690366f, 0.697944f, 0.705540f, 0.713153f, 0.720782f, 0.728427f,
        0.736087f, 0.743758f, 0.751442f, 0.759135f, 0.766837f, 0.774545f, 0.782258f, 0.789974f,
        0.797692f, 0.805409f, 0.813122f, 0.820825f, 0.828515f, 0.836191f, 0.843848f, 0.851476f,
        0.859069f, 0.866624f, 0.874129f, 0.881569f, 0.888942f, 0.896226f, 0.903409f, 0.910473f,
        0.917399f, 0.924168f, 0.930761f, 0.937159f, 0.943348f, 0.949318f, 0.955063f, 0.960587f,
        0.965896f, 0.971003f, 0.975924f, 0.980678f, 0.985282f, 0.989753f, 0.994109f, 0.998364f
    };
    static const float b[] = {
        0.013866f, 0.018570f, 0.024239f, 0.030909f, 0.038558f, 0.046836f, 0.055143f, 0.063460f,
        0.071862f, 0.080282f, 0.088767f, 0.097327f, 0.105930f, 0.114621f, 0.123397f, 0.132232f,
        0.141141f, 0.150164f, 0.159254f, 0.168414f, 0.177642f, 0.186962f, 0.196354f, 0.205799f,
        0.215289f, 0.224813f, 0.234358f, 0.243904f, 0.253430f, 0.262912f, 0.272321f, 0.281624f,
        0.290788f, 0.299776f, 0.308553f, 0.317085f, 0.325338f, 0.333277f, 0.340874f, 0.348111f,
        0.354971f, 0.361447f, 0.367535f, 0.373238f, 0.378563f, 0.383522f, 0.388129f, 0.392400f,
        0.396353f, 0.400007f, 0.403378f, 0.406485f, 0.409345f, 0.411976f, 0.414392f, 0.416608f,
        0.418637f, 0.420491f, 0.422182f, 0.423721f, 0.425116f, 0.426377f, 0.427511f, 0.428524f,
        0.429425f, 0.430217f, 0.430906f, 0.431497f, 0.431994f, 0.432400f, 0.432719f, 0.432955f,
        0.433109f, 0.433183f, 0.433179f, 0.433098f, 0.432943f, 0.432714f, 0.432412f, 0.432039f,
        0.431594f, 0.431080f, 0.430498f, 0.429846f, 0.429125f, 0.428334f, 0.427475f, 0.426548f,
        0.425552f, 0.424488f, 0.423356f, 0.422156f, 0.420887f, 0.419549f, 0.418142f, 0.416667f,
        0.415123f, 0.413511f, 0.411829f, 0.410078f, 0.408258f, 0.406369f, 0.404411f, 0.402385f,
        0.400290f, 0.398125f, 0.395891f, 0.393589f, 0.391219f, 0.388781f, 0.386276f, 0.383704f,
        0.381065f, 0.378359f, 0.375586f, 0.372748f, 0.369846f, 0.366879f, 0.363849f, 0.360757f,
        0.357603f, 0.354388f, 0.351113f, 0.347777f, 0.344383f, 0.340931f, 0.337424f, 0.333861f,
        0.330245f, 0.326576f, 0.322856f, 0.319085f, 0.315266f, 0.311399f, 0.307485f, 0.303526f,
        0.299523f, 0.295477f, 0.291390f, 0.287264f, 0.283099f, 0.278898f, 0.274661f, 0.270390f,
        0.266085f, 0.261750f, 0.257383f, 0.252988f, 0.248564f, 0.244113f, 0.239636f, 0.235133f,
        0.230606f, 0.226055f, 0.221482f, 0.216886f, 0.212268f, 0.207628f, 0.202968f, 0.198286f,
        0.193584f, 0.188860f, 0.184116f, 0.179350f, 0.174563f, 0.169755f, 0.164924f, 0.160070f,
        0.155193f, 0.150292f, 0.145367f, 0.140417f, 0.135440f, 0.130438f, 0.125409f, 0.120354f,
        0.115272f, 0.110164f, 0.105031f, 0.099874f, 0.094695f, 0.089499f, 0.084289f, 0.079073f,
        0.073859f, 0.068659f, 0.063488f, 0.058367f, 0.053324f, 0.048392f, 0.043618f, 0.039050f,
        0.034931f, 0.031409f, 0.028508f, 0.026250f, 0.024661f, 0.023770f, 0.023606f, 0.024202f,
        0.025592f, 0.027814f, 0.030908f, 0.034916f, 0.039886f, 0.045581f, 0.051750f, 0.058329f,
        0.065257f, 0.072489f, 0.079990f, 0.087731f, 0.095694f, 0.103863f, 0.112229f, 0.120785f,
        0.129527f, 0.138453f, 0.147565f, 0.156863f, 0.166353f, 0.176037f, 0.185923f, 0.196018f,
        0.206332f, 0.216877f, 0.227658f, 0.238686f, 0.249972f, 0.261534f, 0.273391f, 0.285546f,
        0.298010f, 0.310820f, 0.323974f, 0.337475f, 0.351369f, 0.365627f, 0.380271f, 0.395289f,
        0.410665f, 0.426373f, 0.442367f, 0.458592f, 0.474970f, 0.491426f, 0.507860f, 0.524203f,
        0.540361f, 0.556275f, 0.571925f, 0.587206f, 0.602154f, 0.616760f, 0.631017f, 0.644924f
    };

    uchar3 host_color_table[256];
    for (int i = 0; i < 256; ++i) {
        host_color_table[i].z = cv::saturate_cast<uchar>(r[i] * 255.0f + 0.5f);  // B 通道
        host_color_table[i].y = cv::saturate_cast<uchar>(g[i] * 255.0f + 0.5f);  // G 通道
        host_color_table[i].x = cv::saturate_cast<uchar>(b[i] * 255.0f + 0.5f);  // R 通道
    }
    cudaMemcpyToSymbol(C_DEVICE_COLOR_MAP, host_color_table, sizeof(uchar3) * 256);
}

// ============================================================================
// TURBO 伪彩色 + P1/P99 分位数归一化（YoloDepthModel 的 CUDA 可视化，全异步、无主机同步）
// 语义对齐 Python depth_to_colormap：
//   去 letterbox(内容区 ROI) → 双线性 resize 回原始分辨率 → 有限值 P1/P99 分位数
//   → clip 归一化 → 灰度 + 查 TURBO 表（表由 cv::applyColorMap 生成，与 Python 调色一致）
// ============================================================================

#define DSTAT_THREADS 256

// BGR 内存布局（与 OpenCV CV_8UC3 一致，wrap 成 cv::Mat 即正确颜色）
__constant__ uchar3 C_DEVICE_TURBO_MAP[256];

void initTurboColorTable() {
    // 用 OpenCV 现成的 applyColorMap 生成 256 色 TURBO 表，保证与 Python cv2 输出一致
    cv::Mat ramp(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
        ramp.at<uchar>(0, i) = static_cast<uchar>(i);
    }
    cv::Mat colored;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 2)
    cv::applyColorMap(ramp, colored, cv::COLORMAP_TURBO);
#else
    cv::applyColorMap(ramp, colored, cv::COLORMAP_JET);  // 旧版 OpenCV 没有 TURBO
#endif
    uchar3 host_table[256];
    for (int i = 0; i < 256; ++i) {
        const cv::Vec3b & c = colored.at<cv::Vec3b>(0, i);
        host_table[i].x     = c[0];  // B
        host_table[i].y     = c[1];  // G
        host_table[i].z     = c[2];  // R
    }
    cudaMemcpyToSymbol(C_DEVICE_TURBO_MAP, host_table, sizeof(uchar3) * 256);
}

// 在 letterbox 内容区 ROI 内做双线性采样（坐标已转 ROI 局部坐标系）
__device__ __forceinline__ float dSampleRoiBilinear(const float * __restrict__ src,
                                                    int   in_w,
                                                    int   roi_x,
                                                    int   roi_y,
                                                    int   roi_w,
                                                    int   roi_h,
                                                    float sx,
                                                    float sy) {
    int         x0 = static_cast<int>(floorf(sx));
    int         y0 = static_cast<int>(floorf(sy));
    const float fx = sx - floorf(sx);
    const float fy = sy - floorf(sy);

    x0           = min(max(x0, 0), roi_w - 1);
    y0           = min(max(y0, 0), roi_h - 1);
    const int x1 = min(x0 + 1, roi_w - 1);
    const int y1 = min(y0 + 1, roi_h - 1);

    const float * row0 = src + static_cast<size_t>(y0 + roi_y) * in_w + roi_x;
    const float * row1 = src + static_cast<size_t>(y1 + roi_y) * in_w + roi_x;
    const float   p00  = row0[x0];
    const float   p10  = row0[x1];
    const float   p01  = row1[x0];
    const float   p11  = row1[x1];
    return p00 * (1.0f - fx) * (1.0f - fy) + p10 * fx * (1.0f - fy) + p01 * (1.0f - fx) * fy +
           p11 * fx * fy;
}

// (1) 内容区 float 深度 → 原始分辨率 float 缓冲（等价 Python remove_letterbox + cv2.resize INTER_LINEAR）
__global__ void depth_resize_float_kernel(const float * __restrict__ src,
                                          int in_w,
                                          int roi_x,
                                          int roi_y,
                                          int roi_w,
                                          int roi_h,
                                          float * __restrict__ dst,
                                          int out_w,
                                          int out_h) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= out_w || oy >= out_h) {
        return;
    }
    const float sx = (ox + 0.5f) * (static_cast<float>(roi_w) / out_w) - 0.5f;
    const float sy = (oy + 0.5f) * (static_cast<float>(roi_h) / out_h) - 0.5f;
    dst[static_cast<size_t>(oy) * out_w + ox] =
        dSampleRoiBilinear(src, in_w, roi_x, roi_y, roi_w, roi_h, sx, sy);
}

// (2) 逐 block 统计有限值 min/max/count（grid = stat_blocks 个 block）
__global__ void depth_stats_kernel(const float * __restrict__ buf,
                                   int n,
                                   float * __restrict__ partial_min,
                                   float * __restrict__ partial_max,
                                   int * __restrict__ partial_count) {
    float local_min = __int_as_float(0x7f800000);  // +inf
    float local_max = __int_as_float(0xff800000);  // -inf
    int   local_cnt = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = buf[i];
        if (isfinite(v)) {
            ++local_cnt;
            local_min = fminf(local_min, v);
            local_max = fmaxf(local_max, v);
        }
    }

    __shared__ float s_min[DSTAT_THREADS];
    __shared__ float s_max[DSTAT_THREADS];
    __shared__ int   s_cnt[DSTAT_THREADS];
    const int        t = threadIdx.x;
    s_min[t]           = local_min;
    s_max[t]           = local_max;
    s_cnt[t]           = local_cnt;
    __syncthreads();

    for (int stride = DSTAT_THREADS / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            s_min[t] = fminf(s_min[t], s_min[t + stride]);
            s_max[t] = fmaxf(s_max[t], s_max[t + stride]);
            s_cnt[t] += s_cnt[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        partial_min[blockIdx.x]   = s_min[0];
        partial_max[blockIdx.x]   = s_max[0];
        partial_count[blockIdx.x] = s_cnt[0];
    }
}

// (3) 汇总各 block 统计 → 全局 min/max（无有限值时为 {0,0}）
__global__ void depth_stats_reduce_kernel(const float * __restrict__ partial_min,
                                          const float * __restrict__ partial_max,
                                          const int * __restrict__ partial_count,
                                          int nblocks,
                                          float * __restrict__ range_2f) {
    float local_min = __int_as_float(0x7f800000);
    float local_max = __int_as_float(0xff800000);
    int   local_cnt = 0;

    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        if (partial_count[b] > 0) {
            local_cnt += partial_count[b];
            local_min = fminf(local_min, partial_min[b]);
            local_max = fmaxf(local_max, partial_max[b]);
        }
    }

    __shared__ float s_min[DSTAT_THREADS];
    __shared__ float s_max[DSTAT_THREADS];
    __shared__ int   s_cnt[DSTAT_THREADS];
    const int        t = threadIdx.x;
    s_min[t]           = local_min;
    s_max[t]           = local_max;
    s_cnt[t]           = local_cnt;
    __syncthreads();

    for (int stride = DSTAT_THREADS / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            s_min[t] = fminf(s_min[t], s_min[t + stride]);
            s_max[t] = fmaxf(s_max[t], s_max[t + stride]);
            s_cnt[t] += s_cnt[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        if (s_cnt[0] > 0) {
            range_2f[0] = s_min[0];
            range_2f[1] = s_max[0];
        } else {
            range_2f[0] = 0.0f;
            range_2f[1] = 0.0f;
        }
    }
}

// (4) 256-bin 直方图（bin 分辨率 ~ (max-min)/256，与 8bit 灰度同精度）
__global__ void depth_hist_kernel(const float * __restrict__ buf,
                                  int n,
                                  const float * __restrict__ range_2f,
                                  int * __restrict__ hist) {
    const float mn  = range_2f[0];
    const float mx  = range_2f[1];
    const float inv = (mx > mn) ? (256.0f / (mx - mn)) : 0.0f;  // 等值场景直接落 bin 0

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = buf[i];
        if (!isfinite(v)) {
            continue;
        }
        int bin = 0;
        if (inv > 0.0f) {
            bin = static_cast<int>((v - mn) * inv);
            bin = min(max(bin, 0), 255);
        }
        atomicAdd(&hist[bin], 1);
    }
}

// 从直方图求第 target 个有序样本的值（线性插值到 bin 内）
__device__ float dHistRankValue(const int * hist,
                                int         total,
                                float       mn,
                                float       bin_width,
                                int         target) {
    if (target <= 0) {
        return mn;
    }
    if (target >= total) {
        return mn + 256.0f * bin_width;
    }
    int cum = 0;
    for (int b = 0; b < 256; ++b) {
        const int c = hist[b];
        if (c == 0) {
            continue;
        }
        if (cum + c >= target) {
            float frac = (target - cum) / static_cast<float>(c);
            frac       = min(max(frac, 0.0f), 1.0f);
            return mn + (static_cast<float>(b) + frac) * bin_width;
        }
        cum += c;
    }
    return mn + 256.0f * bin_width;
}

// (5) 直方图 → P1/P99
__global__ void depth_percentile_kernel(const int * __restrict__ hist,
                                        const float * __restrict__ range_2f,
                                        float * __restrict__ percentiles_2f) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int total = 0;
    for (int b = 0; b < 256; ++b) {
        total += hist[b];
    }
    const float mn    = range_2f[0];
    const float bin_w = (range_2f[1] - mn) / 256.0f;
    if (total == 0) {
        percentiles_2f[0] = 0.0f;
        percentiles_2f[1] = 0.0f;
        return;
    }
    const int t1      = static_cast<int>(total * 0.01f + 0.5f);
    const int t99     = static_cast<int>(total * 0.99f + 0.5f);
    percentiles_2f[0] = dHistRankValue(hist, total, mn, bin_w, t1);
    percentiles_2f[1] = dHistRankValue(hist, total, mn, bin_w, t99);
}

// (6) 归一化 + clip → 灰度 + TURBO 伪彩（直接写原始分辨率输出）
__global__ void depth_colorize_kernel(const float * __restrict__ buf,
                                      int n,
                                      const float * __restrict__ percentiles_2f,
                                      uchar * __restrict__ dst_gray,
                                      uchar3 * __restrict__ dst_color) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float lo  = percentiles_2f[0];
    const float hi  = percentiles_2f[1];
    float       rng = hi - lo;
    if (!(rng > 0.0f)) {
        rng = 1e-6f;  // 与 Python vmax = vmin + 1e-6 兜底一致
    }
    float g          = (buf[i] - lo) / rng;
    g                = fminf(fmaxf(g, 0.0f), 1.0f);
    const uchar gray = static_cast<uchar>(g * 255.0f + 0.5f);  // Python: round(clip)*255

    dst_gray[i]  = gray;
    dst_color[i] = C_DEVICE_TURBO_MAP[gray];
}

void floatDepthColormapResize(const float * src,
                              int           in_w,
                              int           roi_x,
                              int           roi_y,
                              int           roi_w,
                              int           roi_h,
                              float *       stage_float,
                              int           out_w,
                              int           out_h,
                              uchar *       dst_gray,
                              uchar3 *      dst_color,
                              float *       stat_min,
                              float *       stat_max,
                              int *         stat_count,
                              float *       range_2f,
                              int *         hist,
                              float *       percentiles_2f,
                              int           stat_blocks,
                              cudaStream_t  stream) {
    if (roi_w <= 0 || roi_h <= 0 || out_w <= 0 || out_h <= 0 || stat_blocks <= 0) {
        if (dst_gray && out_w > 0 && out_h > 0) {
            cudaMemsetAsync(dst_gray, 0, static_cast<size_t>(out_w) * out_h, stream);
        }
        return;
    }
    const int total        = out_w * out_h;
    const int stat_threads = DSTAT_THREADS;

    // (1) resize
    dim3 blk(32, 8);
    dim3 grid((out_w + 31) >> 5, (out_h + 7) >> 3);
    depth_resize_float_kernel<<<grid, blk, 0, stream>>>(src, in_w, roi_x, roi_y, roi_w, roi_h,
                                                        stage_float, out_w, out_h);
    // (2)(3) min/max/count
    depth_stats_kernel<<<stat_blocks, stat_threads, 0, stream>>>(stage_float, total, stat_min,
                                                                 stat_max, stat_count);
    depth_stats_reduce_kernel<<<1, stat_threads, 0, stream>>>(stat_min, stat_max, stat_count,
                                                              stat_blocks, range_2f);
    // (4)(5) P1/P99
    cudaMemsetAsync(hist, 0, 256 * sizeof(int), stream);
    depth_hist_kernel<<<stat_blocks, stat_threads, 0, stream>>>(stage_float, total, range_2f, hist);
    depth_percentile_kernel<<<1, 1, 0, stream>>>(hist, range_2f, percentiles_2f);
    // (6) 灰度 + TURBO
    depth_colorize_kernel<<<(total + stat_threads - 1) / stat_threads, stat_threads, 0, stream>>>(
        stage_float, total, percentiles_2f, dst_gray, dst_color);
}
