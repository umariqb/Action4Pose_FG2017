#include "armaknn.h"
#include <armadillo>

using namespace arma;
using namespace std;

#include <queue>
#include <vector>
#include <algorithm>
//#include <omp.h>


/** This method computes the exact k-nearest neighbors (only for L2-norm) for a
 *  given set of  queries stored in a matrix q (each column is one vector) to a
 *  given set of reference vectors stored in matrix v (one vector per colum).
 *
 *  This implementation subdivides the problem into smaller block (cache
 *  friendly) and ensures that no extremely large allocations are made, i.e. if
 *  the number of query and reference vectors grows huge which results in a
 *  very big distance matrix.
 *
 *  The inlined constant KNN_BLOCKSIZE defines the size of the subblocks.
 *
 *  Thanks to the blocking mechanism and the full matrix formulation and the
 *  inline heap search of the k nearest neighbors, this method is lightning fast
 *
 *  Regarding speed, it is difficult to parallelize this code using openmp (but
 *  it is very easy to compute many knn at the same time.
 *
 *  \param   q     The query vectors (one vector per column)
 *  \param   v     The reference vectors (one vector per column)
 *  \param   d     The distance matrix of size k x number_of_queries
 *                 containing the k smallest distances
 *  \param   id    The indices of size k x number_of_queries
 *                 containing the indices of the reference vectors having the
 *                 smallest distances
 *  \param   k     The number of nearest neighbors
 *
 *  \author  Dr. Christian Wengert
 *           christian.wengert@ieee.org
 */
void knn_L2( const fmat &queryVectors,
             const fmat &referenceVectors,
                   fmat &distanceMap,
                   umat &indexMap,
                   const unsigned int k)
{
  const unsigned int uiNQueries = queryVectors.n_cols;
  const unsigned int uiNBase = referenceVectors.n_cols;

  const unsigned int KNN_BLOCKSIZE=256;//Cache dependent
  //however, take care of actual block size and adapt accordingly
  const unsigned int uiBlockSize1 = min(uiNQueries, KNN_BLOCKSIZE);
  const unsigned int uiBlockSize2 = min(uiNBase, KNN_BLOCKSIZE);

  //the k-nearest neighbors are stored in a vector of priority queues (max-heaps)
  //of pairs<distance, index to codeword>
  //Could be reused to avoid re-allocations (minor optimization)
  vector<priority_queue<pair<float, unsigned int> > > knnDistancesHeap(uiNQueries);

  //prefill heaps with nonsense (=large values) to ensure a fixed size of k
  //elements per heap
//#pragma omp parallel for shared(knnDistancesHeap)
  for (unsigned int i=0;i<uiNQueries;i++  )
  {
    for(unsigned int kk=0;kk<k;kk++)
    {
      knnDistancesHeap[i].push(make_pair(numeric_limits<float>::max(), numeric_limits<int>::max()));  //add some dummy values
    }
  }

  //preallocate (maybe faster?)
  fmat a,aa,b,bb,ab;

  //go over query vector blocks
//#pragma omp parallel for shared(knnDistancesHeap)
  for(unsigned int i=0; i<uiNQueries; i+=uiBlockSize1)
  {
    //compute indices for sub-matrices
    unsigned int idx_a1 = i;
    unsigned int idx_a2 = min( uiNQueries-1, i+uiBlockSize1-1);

    //precompute
    a  = queryVectors.cols(  idx_a1, idx_a2  );
    aa = sum(square(a));

    //go over reference vector blocks
    for(unsigned int j=0; j<uiNBase; j+=uiBlockSize2)
    {
      //compute indices for sub-matrices
      unsigned int idx_b1 = j;
      unsigned int idx_b2 = min( uiNBase-1, j+uiBlockSize2-1);

      //precompute
      b  = referenceVectors.cols(  idx_b1, idx_b2  );
      bb = sum(square(b));
      ab = trans(a)*b;

      //partial distance matrix
      distanceMap = repmat(trans(aa),1, bb.n_cols) + repmat(bb, aa.n_cols, 1) - 2*ab;

      //fill the heaps
      for (unsigned int r=0;r<distanceMap.n_rows;r++)
      {
        for (unsigned int s=0;s<distanceMap.n_cols;s++)
        {
          const unsigned int n = idx_a1+r;
          if( knnDistancesHeap[n].top().first > distanceMap.at(r,s) )
          {
            //add to queue
            knnDistancesHeap[n].push( make_pair(distanceMap.at(r,s), idx_b1+s) );
            //remove an element
            knnDistancesHeap[n].pop();
          }
        }
      }
    }
  }


  //allocate: the set_size call only reallocates if required
  distanceMap.set_size(k, uiNQueries);
  indexMap.set_size(k, uiNQueries);
  //copy distances and ids
//#pragma omp parallel for shared(knnDistancesHeap, distanceMap, indexMap)
  for (unsigned int i=0;i<uiNQueries;i++  )
  {
    for(unsigned int kk=0;kk<k;kk++)
    {
      unsigned int index = k-kk-1;
      distanceMap.at(index, i) = knnDistancesHeap[i].top().first;
      indexMap.at(index, i) = knnDistancesHeap[i].top().second;
      knnDistancesHeap[i].pop();
    }
  }
}
