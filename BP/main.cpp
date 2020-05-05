#include <iostream>
#include <cstring>
#include <functional>
#include <fstream>
#include <sstream>
#include <ctime>
#include <tuple>
#include <iterator>
#include <random>
#include <initializer_list>
using namespace std;

//*******************************************************************************
//                                      矩阵类                                   *
//*******************************************************************************
struct Matrix {
//实例
public:
    size_t rows, cols;
    double *matrix;
//构造函数
public:
    Matrix() : rows(0), cols(0), matrix(nullptr) { }
    ~Matrix() { if (matrix) delete []matrix; }
    //有参构造函数
    Matrix(size_t rows, size_t cols): rows(rows), cols(cols), matrix(new double[rows * cols]) {}
    Matrix(size_t rows, size_t cols, double value): rows(rows), cols(cols), matrix(new double[rows * cols]){
        for(size_t i = 0;i < rows * cols; ++i) matrix[i] = value;
    }
    Matrix(size_t rows, size_t cols, initializer_list<double> array): rows(rows), cols(cols), matrix(new double[rows * cols]){
        size_t i = 0;
        for(auto iter = array.begin(); iter != array.end(); ++iter) matrix[i++] = *iter;
    }
    template<typename InputIterator>
    Matrix(size_t rows, size_t cols, InputIterator first, InputIterator last): rows(rows), cols(cols), matrix(new double[rows * cols]){
        size_t i = 0;
        for(auto iter = first; iter != last; ++iter) matrix[i++] = *iter;
    }
    //复制构造函数（左值）
    Matrix(const Matrix &orig): rows(orig.rows), cols(orig.cols), matrix(new double[rows * cols]){
        memcpy(matrix, orig.matrix, sizeof(double) * rows * cols);
    }
    //复制构造函数（右值）
    Matrix(Matrix &&orig): rows(orig.rows), cols(orig.cols), matrix(orig.matrix){
        orig.rows = orig.cols = 0;
        orig.matrix = nullptr;
    }
//方法
public:
    double &at(int row, int col) { return matrix[row * cols + col]; }
    double at(int row, int col) const { return matrix[row * cols + col]; }
    //重载  =
    //左值
    Matrix &operator = (const Matrix &orig) {
        if (this != &orig) {
            if (matrix) delete []matrix;
            rows = orig.rows; cols = orig.cols; matrix = new double[rows * cols];
            memcpy(matrix, orig.matrix, sizeof(double) * rows * cols);
        }
        return *this;
    }
    //重载  =
    //右值
    Matrix &operator = (Matrix &&orig) {
        if (this != &orig) {
            if (matrix) delete []matrix;
            rows = orig.rows; cols = orig.cols; matrix = orig.matrix;
            orig.rows = orig.cols = 0; orig.matrix = nullptr;
        }
        return *this;
    }
    size_t size() const { return rows * cols; }
    double magnitude() const {
        double sum = 0;
        for (size_t i = 0; i < size(); ++i)
            sum += matrix[i] * matrix[i];
        return sqrt(sum);
    }
    Matrix submatrix(size_t startrow, size_t endrow, size_t startcol, size_t endcol) const {
        Matrix matrix(endrow - startrow, endcol - startcol);
        for (size_t i = startrow; i < endrow; ++i)
            for (size_t j = startcol; j < endcol; ++j)
                matrix.at(i - startrow, j - startcol) = this->at(i, j);
        return matrix;
    }
    //Matrix作用域
    static Matrix transpose(const Matrix &matrix);//转置
    static Matrix hadamard(const Matrix &lhs, const Matrix &rhs);//对应位置的值相乘，结果大小与左矩阵相同
    static Matrix kronecker(const Matrix &lhs, const Matrix &rhs);//克罗内克积(张量积的特殊形式)
    static Matrix horcat(const Matrix &lhs, const Matrix &rhs);//合并矩阵，结果大小的行数与左矩阵相同
    static Matrix apply(const Matrix &matrix, function<double(double)> func);//对矩阵的每一个元素做function
};

//操作符重载
ostream &operator << (ostream &out, const Matrix &matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols - 1; ++j)
            out << matrix.at(i, j) << "\t";
        if (matrix.cols > 0) out << matrix.at(i, matrix.cols - 1);
        out << '\n';
    }
    return out;
}
istream &operator >> (istream &in, Matrix &matrix) {
    for (size_t i = 0; i < matrix.rows; ++i)
        for (size_t j = 0; j < matrix.cols; ++j)
            in >> matrix.at(i, j);
    return in;
}
inline Matrix &operator += (Matrix &lhs, const Matrix &rhs) {
    //assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols);
    for (size_t i = 0; i < lhs.size(); ++i) lhs.matrix[i] += rhs.matrix[i]; return lhs;
}
inline Matrix &operator -= (Matrix &lhs, const Matrix &rhs) {
    //assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols);
    for (size_t i = 0; i < lhs.size(); ++i) lhs.matrix[i] -= rhs.matrix[i]; return lhs;
}
inline Matrix operator + (const Matrix &lhs, const Matrix &rhs)
{ Matrix result(lhs); result += rhs; return result; }
inline Matrix operator - (const Matrix &lhs, const Matrix &rhs)
{ Matrix result(lhs); result -= rhs; return result; }
inline Matrix &operator *= (Matrix &lhs, double num)
{ for (size_t i = 0; i < lhs.size(); ++i) lhs.matrix[i] *= num; return lhs; }
inline Matrix operator * (const Matrix &lhs, double num)
{ Matrix result(lhs); result *= num; return result; }
inline Matrix operator * (double num, const Matrix &rhs)
{ return rhs * num; }
Matrix &operator *= (Matrix &lhs, const Matrix &rhs) {
    //assert(lhs.cols == rhs.rows);
    double *temp = new double[lhs.rows * rhs.cols];
    memset(temp, 0, sizeof(double) * lhs.rows * rhs.cols);
    for (size_t row = 0; row < lhs.rows; ++row)
        for (size_t col = 0; col < rhs.cols; ++col)
            for (size_t inner = 0; inner < lhs.cols; ++inner)
                temp[row * rhs.cols + col] += lhs.at(row, inner) * rhs.at(inner, col);
    lhs.cols = rhs.cols;
    if (lhs.matrix) delete []lhs.matrix;
    lhs.matrix = temp;
    return lhs;
}
inline Matrix operator * (const Matrix &lhs, const Matrix &rhs)
{ Matrix result(lhs); result *= rhs; return result; }

//作用域内函数的实现
Matrix Matrix::transpose(const Matrix &matrix) {
    Matrix result(matrix.cols, matrix.rows);
    for (size_t i = 0; i < matrix.rows; ++i)
        for (size_t j = 0; j < matrix.cols; ++j)
            result.at(j, i) = matrix.at(i, j);
    return result;
}
Matrix Matrix::hadamard(const Matrix &lhs, const Matrix &rhs) {
    //assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols);
    Matrix result(lhs);
    for (size_t i = 0; i < lhs.rows; ++i)
        for (size_t j = 0; j < lhs.cols; ++j)
            result.at(i, j) *= rhs.at(i, j);
    return result;
}
Matrix Matrix::kronecker(const Matrix &lhs, const Matrix &rhs) {
    Matrix result(lhs.rows * rhs.rows, lhs.cols * rhs.cols);
    for (size_t m = 0; m < lhs.rows; ++m)
        for (size_t n = 0; n < lhs.cols; ++n)
            for (size_t p = 0; p < rhs.rows; ++p)
                for (size_t q = 0; q < rhs.cols; ++q)
                    result.at(m * rhs.rows + p, n * rhs.cols + q) = lhs.at(m, n) * rhs.at(p, q);
    return result;
}
Matrix Matrix::horcat(const Matrix &lhs, const Matrix &rhs) {
    //assert(lhs.rows == rhs.rows);
    Matrix result(lhs.rows, lhs.cols + rhs.cols);
    for (size_t i = 0; i < result.rows; ++i)
        for (size_t j = 0; j < result.cols; ++j)
            result.at(i, j) = j < lhs.cols ? lhs.at(i, j) : rhs.at(i, j - lhs.cols);
    return result;
}
Matrix Matrix::apply(const Matrix &matrix, function<double(double)> func) {
    Matrix result(matrix.rows, matrix.cols);
    for (size_t i = 0; i < matrix.rows; ++i)
        for (size_t j = 0; j < matrix.cols; ++j)
            result.at(i, j) = func(matrix.at(i, j));
    return result;
}

//*******************************************************************************
//                                      神经网络层类                              *
//*******************************************************************************
//激活函数
double activate(double x) {
    return 1.0 / (1.0 + exp(-x));
}
//激活函数的导数
double activate_diff(double y) {
    return y * (1.0 - y);
}

struct NeuralLayer {
//实例
public:
    Matrix weights;//最后一行为权值
//构造函数
public:
    NeuralLayer() {}
    ~NeuralLayer() {}

    //有参构造函数

    //输入值加一行是权值行（对应输入矩阵加一列全为1的值）
    NeuralLayer(size_t n_input, size_t n_output): weights(n_input + 1, n_output) {
        initialise_random();
    }
    NeuralLayer(size_t n_input, size_t n_output, initializer_list<double> data): weights(n_input + 1, n_output, data) {}
    template<typename InputIterator>
    NeuralLayer(size_t n_input, size_t n_output, InputIterator first, InputIterator last): weights(n_input + 1, n_output, first, last) {}
    //复制构造函数 左值
    NeuralLayer(const NeuralLayer &orig): weights(orig.weights) {}
    //复制构造函数 右值
    NeuralLayer(NeuralLayer &&orig): weights(move(orig.weights)) {}
    NeuralLayer &operator = (const NeuralLayer &orig) { weights = orig.weights; return *this; }
    NeuralLayer &operator = (NeuralLayer &&orig) { weights = move(orig.weights); return *this; }
    //产生一列全为1的值
    static Matrix generate_bias(size_t row_count) {
        return Matrix(row_count, 1, 1.0);
    }
    //初始化矩阵
    void initialise_random() {
        static random_device rd;
        static mt19937 e2(rd());
        uniform_real_distribution<> dist(-0.5, 0.5);
        double factor = 0.7 * pow(weights.cols, 1.0 / weights.cols);
        //给矩阵赋随机值
        for (size_t i = 0; i < weights.rows; ++i)
            for (size_t j = 0; j < weights.cols; ++j)
                weights.at(i, j) = dist(e2);
        //对随机值加工
        double magnitude = weights.magnitude();
        for (size_t i = 0; i < weights.rows; ++i)
            for (size_t j = 0; j < weights.cols; ++j)
                weights.at(i, j) *= factor / magnitude;
    }
    //前向传播
    //返回 （输入矩阵+一列全为1的值）*权重矩阵，对它们的结果矩阵中的每一个值激活
    Matrix feedforward(const Matrix &inputs) {
        return Matrix::apply(Matrix::horcat(inputs, generate_bias(inputs.rows)) * weights, [](double x){ return activate(x); });
    }
    //反向传播
    //获得每一层的delta，然后根据每一层的delta得出下一层的err，对权重矩阵进行调整
    Matrix backpropagation(const Matrix &input, const Matrix &output, const Matrix &error, double rate, int index = 0) {
        Matrix delta = Matrix::hadamard(Matrix::apply(output, [](double x) { return activate_diff(x); }), error), newerror
        //因为将w矩阵放在了右边，所以要进行转置
        if (index) newerror = (delta * Matrix::transpose(weights)).submatrix(0, 1, 0, weights.rows - 1);
        weights += rate * Matrix::kronecker(Matrix::transpose(Matrix::horcat(input, generate_bias(1))), delta);
        return newerror;
    }
};

int main() {
    //模拟一个二输入，三个结点的隐含层，一个结点的输出层

    //输入层
    initializer_list<double> ls_in{0.1,0.2,
                                   0.3,0.4};
    initializer_list<double> ls_out{0.3,
                                    0.7};
    Matrix input(2,2,ls_in);
    //样本输出
    Matrix data_out(2,1,ls_out);
    //隐含层
    NeuralLayer hide(2,3);
    //输出层
    NeuralLayer outLay(3,1);

    //正向传播
    auto hide_out=hide.feedforward(input);
    cout<<hide_out<<endl;
    auto output=outLay.feedforward(hide_out);
    cout<<output<<endl;

    //误差反向传播

    return 0;
}