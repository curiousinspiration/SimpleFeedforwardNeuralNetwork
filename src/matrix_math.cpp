/*
 * Matrix Math Implementation
 *
 */

#include "neural/math/matrix_math.h"

#include <sstream>

using namespace std;

namespace neural
{

TMatrix MatrixMath::Multiply(const TMatrix& a_lhs, const TMatrix& a_rhs)
{
    // Check to make sure the inner dimensions of our matrices line up
    if (a_lhs.at(0).size() != a_rhs.size())
    {
        stringstream l_ss;
        l_ss << "Inner dimensions of matrices must match "
             << a_lhs.at(0).size() << " != " << a_rhs.size();
        throw(runtime_error(l_ss.str()));
    }

    // initialize our return matrix with the correct shape,
    // ie the outer sizes of our inputs and rhs
    TMatrix l_ret;
    for (size_t i = 0; i < a_lhs.size(); ++i)
    {
        // add an empty vector for each row
        l_ret.push_back(vector<float>());
        for (size_t j = 0; j < a_rhs.at(0).size(); ++j)
        {
            // fill the columns with 0s
            l_ret.at(i).push_back(0);
        }
    }

    // walk over rows of lhs
    for (size_t i = 0; i < a_lhs.size(); ++i)
    {
        // walk over cols of rhs
        for (size_t j = 0; j < a_rhs.at(0).size(); ++j)
        {
            // multiply element from row i in lhs by col j in weight
            // and sum up the values into i,j
            for (size_t k = 0; k < a_rhs.size(); ++k)
            {
                // LOG(INFO) << "i " << i << " j " << j << " k " << a_lhs.at(i).at(k) << " * " << a_rhs.at(k).at(j) << endl;
                l_ret.at(i).at(j) += (a_lhs.at(i).at(k) * a_rhs.at(k).at(j));
            }
        }
    }
    return l_ret;
}

TMatrix MatrixMath::Transpose(const TMatrix& a_mat)
{
    size_t x = a_mat.at(0).size();
    size_t y = a_mat.size();
    
    TMatrix l_ret;
    for (size_t i = 0; i < x; ++i)
    {
        l_ret.push_back(vector<float>());
        for (size_t j = 0; j < y; ++j)
        {
            l_ret.at(i).push_back(a_mat.at(j).at(i));
        }
    }

    return l_ret;
}

TMatrix MatrixMath::AddCol(const TMatrix& a_mat, float a_val)
{
    TMatrix l_ret = a_mat;
    for (size_t i = 0; i < l_ret.size(); ++i)
    {
        l_ret.at(i).push_back(a_val);
    }
    return l_ret;
}

TMatrix MatrixMath::RemoveCol(const TMatrix& a_mat)
{
    TMatrix l_ret = a_mat;
    for (size_t i = 0; i < l_ret.size(); ++i)
    {
        l_ret.at(i).pop_back();
    }
    return l_ret;
}

} // namespace neural
