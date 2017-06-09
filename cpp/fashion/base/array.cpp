#include "array.hpp"

#include <fstream>
#include <cstring>

namespace fashion
{
    Array<float> padarray(const Array<float>& source, unsigned int padx, unsigned int pady)
    {
        unsigned int newsize_x = source.size(0) + 2 * padx;
        unsigned int newsize_y = source.size(1) + 2 * pady;
        unsigned int newsize_z = source.size(2);

        Array<float> destination(newsize_x, newsize_y, newsize_z);

        destination.zero();

        // copy array
        for (unsigned int z = 0; z < source.size(2); ++z)
            for (unsigned int y = 0; y < source.size(1); ++y)
                std::memcpy(
                        destination.values() + z * destination.step(2) + (y + pady) * destination.step(1) + padx,
                        source.values() + z * source.step(2) + y * source.step(1),
                        source.size(0) * sizeof(float));

        return destination;
    }

    template <>
    void readArray<float>(Array<float>& array, const std::string& filename)
    {
        std::ifstream file(filename.c_str(), std::ifstream::in);

        if (file.good())
        {
            for (unsigned int b = 0; b < array.size(1); ++b)
                for (unsigned int c = 0; c < array.size(2); ++c)
                    for (unsigned int a = 0; a < array.size(0); ++a)
                        file >> array(a, b, c);
        }
        else
        {
            std::cout << "Error: Couldn't open \"" << filename << "\"" << std::endl;
        }

        file.close();
    }
}
