/*
 * Matrix Math Definition
 *
 */

#pragma once

#include "neural/typedefs.h"

namespace neural
{

class MatrixMath
{
public:
    static TMatrix Multiply(const TMatrix& a_lhs, const TMatrix& a_rhs);
    static TMatrix Transpose(const TMatrix& a_mat);
};

} // namespace neural
