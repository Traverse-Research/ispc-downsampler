#pragma once
inline varying float length(varying float<3> v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline varying float length(varying float<4> v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

inline varying float<3> normalize(varying float<3> v) {
    float length = length(v);
    return v / length;
}

inline varying float<4> normalize(varying float<4> v) {
    float length = length(v);
    return v / length;
}
