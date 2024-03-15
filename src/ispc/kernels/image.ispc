#ifndef ISPC_IMAGE
#define ISPC_IMAGE
struct Image {
    uniform uint8* data;
    uniform int<2> size;
};
#endif
