#ifndef _ARRAY_H__
#define _ARRAY_H__

#include <vector>
#include <iostream>

#include <boost/shared_array.hpp>
#include <opencv2/opencv.hpp>

namespace fashion
{
    // Declaration
    template <class T>
    class Array
    {
        public:
            static const size_t MAX_DIMENSIONS = 3;

            inline Array();
            inline Array(unsigned int size1, T* values = 0);
            inline Array(unsigned int size1, unsigned int size2, T* values = 0);
            inline Array(unsigned int size1, unsigned int size2, unsigned int size3, T* values = 0);
            inline Array(const unsigned int* size, T* values = 0);

            inline Array(const Array<T>& other);
            inline const Array<T>& operator=(const Array<T>& other);

            inline void reset();
            inline void set(unsigned int size1, T* values = 0);
            inline void set(unsigned int size1, unsigned int size2, T* values = 0);
            inline void set(unsigned int size1, unsigned int size2, unsigned int size3, T* values = 0);
            inline void set(const unsigned int* size, T* values = 0);

            inline size_t dimensions() const;
            inline unsigned int size(size_t dimension) const;
            inline const unsigned int* size() const;
            inline unsigned int step(size_t dimension) const;
            inline const unsigned int* step() const;
            inline unsigned int length() const;

            inline T* values();
            inline const T* values() const;
            inline cv::Mat& matrix();

            inline T operator[](unsigned int pos) const;
            inline T& operator[](unsigned int pos);

            inline T operator()(unsigned int pos1) const;
            inline T operator()(unsigned int pos1, unsigned int pos2) const;
            inline T operator()(unsigned int pos1, unsigned int pos2, unsigned int pos3) const;
            inline T& operator()(unsigned int pos1);
            inline T& operator()(unsigned int pos1, unsigned int pos2);
            inline T& operator()(unsigned int pos1, unsigned int pos2, unsigned int pos3);

            void zero();

            void printDimensions() const;
            void printValues() const;

            void read(const std::string& filename);

            Array<T>& operator+=(const Array<T>& other);

            template <class O>
            Array<O> convertTo() const;

        private:
            inline void createMatrix(int type);

            unsigned int sizes_[MAX_DIMENSIONS];
            unsigned int steps_[MAX_DIMENSIONS];

            T* values_;
            cv::Mat matrix_;
            boost::shared_array<T> shared_ptr_;
    };

    // Specialized functions
    Array<float> padarray(const Array<float>& features, unsigned int padx, unsigned int pady);

    template <class T>
    void readArray(Array<T>& array, const std::string& filename);

    // Implementation
    template <class T>
    Array<T>::Array()
    {
        this->reset();
    }

    template <class T>
    Array<T>::Array(unsigned int size1, T* values)
    {
        this->set(size1, values);
    }

    template <class T>
    Array<T>::Array(unsigned int size1, unsigned int size2, T* values)
    {
        this->set(size1, size2, values);
    }

    template <class T>
    Array<T>::Array(unsigned int size1, unsigned int size2, unsigned int size3, T* values)
    {
        this->set(size1, size2, size3, values);
    }

    template <class T>
    Array<T>::Array(const unsigned int* size, T* values)
    {
        this->set(size, values);
    }

    template <class T>
    Array<T>::Array(const Array<T>& other)
    {
        this->values_ = other.values_;
        this->matrix_ = other.matrix_;
        this->shared_ptr_ = other.shared_ptr_;

        this->sizes_[0] = other.sizes_[0];
        this->sizes_[1] = other.sizes_[1];
        this->sizes_[2] = other.sizes_[2];

        this->steps_[0] = other.steps_[0];
        this->steps_[1] = other.steps_[1];
        this->steps_[2] = other.steps_[2];
    }

    template <class T>
    const Array<T>& Array<T>::operator=(const Array<T>& other)
    {
        this->values_ = other.values_;
        this->matrix_ = other.matrix_;
        this->shared_ptr_ = other.shared_ptr_;

        this->sizes_[0] = other.sizes_[0];
        this->sizes_[1] = other.sizes_[1];
        this->sizes_[2] = other.sizes_[2];

        this->steps_[0] = other.steps_[0];
        this->steps_[1] = other.steps_[1];
        this->steps_[2] = other.steps_[2];

        return *this;
    }

    template <class T>
    void Array<T>::reset()
    {
        this->values_ = 0;
        this->shared_ptr_.reset();

        this->sizes_[0] = 0;
        this->sizes_[1] = 1;
        this->sizes_[2] = 1;

        this->steps_[0] = 0;
        this->steps_[1] = 0;
        this->steps_[2] = 0;
    }

    template <class T>
    void Array<T>::set(unsigned int size1, T* values)
    {
        if (values == 0)
            values = new T[size1];

        this->values_ = values;
        this->shared_ptr_.reset(values);

        this->sizes_[0] = size1;
        this->sizes_[1] = 1;
        this->sizes_[2] = 1;

        this->steps_[0] = 1;
        this->steps_[1] = 0;
        this->steps_[2] = 0;
    }

    template <class T>
    void Array<T>::set(unsigned int size1, unsigned int size2, T* values)
    {
        if (values == 0)
            values = new T[size1 * size2];

        this->values_ = values;
        this->matrix_ = cv::Mat(size1, size2, cv::DataType<T>::type, values);
        this->shared_ptr_.reset(values);

        this->sizes_[0] = size1;
        this->sizes_[1] = size2;
        this->sizes_[2] = 1;

        this->steps_[0] = 1;
        this->steps_[1] = size1;
        this->steps_[2] = size1;
    }

    template <class T>
    void Array<T>::set(unsigned int size1, unsigned int size2, unsigned int size3, T* values)
    {
        if (values == 0)
            values = new T[size1 * size2 * size3];

        this->values_ = values;
        this->shared_ptr_.reset(values);

        this->sizes_[0] = size1;
        this->sizes_[1] = size2;
        this->sizes_[2] = size3;

        this->steps_[0] = 1;
        this->steps_[1] = size1;
        this->steps_[2] = size1 * size2;
    }

    template <class T>
    void Array<T>::set(const unsigned int* size, T* values)
    {
        this->sizes_[0] = size[0];
        this->sizes_[1] = size[1];
        this->sizes_[2] = size[2];

        switch (this->dimensions())
        {
            case 1: this->set(size[0], values); break;
            case 2: this->set(size[0], size[1], values); break;
            case 3: this->set(size[0], size[1], size[2], values); break;
            default:
                this->values_ = values;
                this->shared_ptr_.reset(values);
                break;
        }
    }

    template <class T>
    size_t Array<T>::dimensions() const
    {
        if (this->sizes_[2] > 1)
            return 3;
        else if (this->sizes_[1] > 1)
            return 2;
        else if (this->sizes_[0] > 0)
            return 1;
        else
            return 0;
    }

    template <class T>
    unsigned int Array<T>::size(size_t dimension) const
    {
        return this->sizes_[dimension];
    }

    template <class T>
    const unsigned int* Array<T>::size() const
    {
        return this->sizes_;
    }

    template <class T>
    unsigned int Array<T>::step(size_t dimension) const
    {
        return this->steps_[dimension];
    }

    template <class T>
    const unsigned int* Array<T>::step() const
    {
        return this->steps_;
    }

    template <class T>
    unsigned int Array<T>::length() const
    {
        unsigned int size = 1;
        for (size_t i = 0; i < MAX_DIMENSIONS; ++i)
            size *= this->sizes_[i];
        return size;
    }

    template <class T>
    T* Array<T>::values()
    {
        return this->values_;
    }

    template <class T>
    const T* Array<T>::values() const
    {
        return this->values_;
    }

    template <class T>
    cv::Mat& Array<T>::matrix()
    {
        return this->matrix_;
    }

    template <class T>
    T Array<T>::operator[](unsigned int pos) const
    {
        return this->values_[pos];
    }

    template <class T>
    T& Array<T>::operator[](unsigned int pos)
    {
        return this->values_[pos];
    }

    template <class T>
    T Array<T>::operator()(unsigned int pos1) const
    {
        return this->values_[pos1];
    }

    template <class T>
    T Array<T>::operator()(unsigned int pos1, unsigned int pos2) const
    {
        return this->values_[pos2 * this->steps_[1] + pos1];
    }

    template <class T>
    T Array<T>::operator()(unsigned int pos1, unsigned int pos2, unsigned int pos3) const
    {
        return this->values_[pos3 * this->steps_[2] + pos2 * this->steps_[1] + pos1];
    }

    template <class T>
    T& Array<T>::operator()(unsigned int pos1)
    {
        return this->values_[pos1];
    }

    template <class T>
    T& Array<T>::operator()(unsigned int pos1, unsigned int pos2)
    {
        return this->values_[pos2 * this->steps_[1] + pos1];
    }

    template <class T>
    T& Array<T>::operator()(unsigned int pos1, unsigned int pos2, unsigned int pos3)
    {
        return this->values_[pos3 * this->steps_[2] + pos2 * this->steps_[1] + pos1];
    }

    template <class T>
    void Array<T>::zero()
    {
        // write zeros in x dimension
        for (unsigned int x = 0; x < this->sizes_[0]; ++x)
            this->values_[x] = 0;

        // write zeros in y dimension
        for (unsigned int y = 1; y < this->sizes_[1]; ++y)
            memcpy(this->values_ + y * this->steps_[1], this->values_, this->steps_[1] * sizeof(float));

        // write zeros in z dimension
        for (unsigned int z = 1; z < this->sizes_[2]; ++z)
            memcpy(this->values_ + z * this->steps_[2], this->values_, this->steps_[2] * sizeof(float));
    }

    template <class T>
    void Array<T>::printDimensions() const
    {
        switch (this->dimensions())
        {
            case 1: std::cout << this->sizes_[0] << std::endl; break;
            case 2: std::cout << this->sizes_[0] << 'x' << this->sizes_[1] << std::endl; break;
            case 3: std::cout << this->sizes_[0] << 'x' << this->sizes_[1] << 'x' << this->sizes_[2] << std::endl; break;
        }
    }

    template <class T>
    void Array<T>::printValues() const
    {
        std::streamsize precision = std::cout.precision(4);

        for (unsigned int c = 0; c < this->sizes_[2]; ++c)
        {
            std::cout << c << ':' << std::endl;
            for (unsigned int b = 0; b < this->sizes_[1]; ++b)
            {
                for (unsigned int a = 0; a < this->sizes_[0]; ++a)
                {
                    std::cout << std::fixed << this->operator()(a, b, c) << '\t';
                }

                std::cout << std::endl;
            }
        }

        std::cout.precision(precision);
    }

    template <class T>
    void Array<T>::read(const std::string& filename)
    {
        readArray(*this, filename);
    }

    template <class T>
    void readArray(Array<T>&, const std::string&); // unspecialized implementation is not defined

    template <>
    void readArray<float>(Array<float>& array, const std::string& filename);

    template <class T>
    Array<T>& Array<T>::operator+=(const Array<T>& other)
    {
        if (this->length() == 0)
        {
            *this = other;
        }
        else
        {
            unsigned int new_size = other.length() + this->length();

            Array<T> new_array(new_size);

            memcpy(new_array.values(), this->values(), this->length() * sizeof(T));
            memcpy(new_array.values() + this->length(), other.values(), other.length() * sizeof(T));

            *this = new_array;
        }

        return *this;
    }

    template <class T>
    template <class O>
    Array<O> Array<T>::convertTo() const
    {
        Array<O> other(this->size());
        for (size_t i = 0; i < this->length(); ++i)
            other[i] = (*this)[i];
        return other;
    }
}

#endif /* _ARRAY_H__ */
