
varying float length(varying float<3> v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

varying float length(varying float<4> v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

varying float<3> normalize(varying float<3> v) {
    float length = length(v);
    return v / length;
}

varying float<4> normalize(varying float<4> v) {
    float length = length(v);
    return v / length;
}
