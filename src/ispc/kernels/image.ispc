#ifndef ISPC_IMAGE
#define ISPC_IMAGE
struct Image {
    uniform uint8* data;
    uniform int<2> size;
    // Bytes from one row to the next.
    uniform uint32 row_pitch;
};
#endif
