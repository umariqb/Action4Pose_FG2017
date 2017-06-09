/**
 * @file DereferencingIterator.h
 * @author boslu
 * @date 17.11.2009
 *
 */

#ifndef DATA_STRUCTURE__UTILS_DEREFERENCINGITERATOR_H_
#define DATA_STRUCTURE__UTILS_DEREFERENCINGITERATOR_H_

// standard library includes

// external library includes
#include <boost/operators.hpp>

// local includes
namespace utils {
namespace data_structure
{

template<class T, class W>
class DereferencingIterator:
    public boost::random_access_iterator_helper<DereferencingIterator<T, W>, T>

{
  typedef DereferencingIterator self_type;
  typedef W                     wrapped_iterator;

public:
  typedef T                                         value_type;
  typedef value_type&                               reference;
  typedef typename wrapped_iterator::value_type     pointer;
  typedef std::ptrdiff_t                            difference_type;

  DereferencingIterator() {

  }

  // assignment operator
  self_type& operator=(const self_type& i) {
    mIterator = i.mIterator;
    return *this;
  }

  //copy constructor. allows iterator to const_iterator
  template<class Type, class Wrapped>
  DereferencingIterator(const DereferencingIterator<Type, Wrapped>& i)
      : mIterator(i.base()) {
  }

  // wrapped converstion
  explicit DereferencingIterator(const wrapped_iterator& i)
      : mIterator(i) {

  }

  self_type& operator++() {
    ++mIterator;
    return *this;
  }

  self_type& operator--() {
    --mIterator;
    return *this;
  }

  reference operator*() const {
    return **mIterator;
  }

  // exposes pointer
  pointer operator->() const {
    return this->getPtr();
  }

  pointer getPtr() const {
    return *mIterator;
  }

  self_type& operator+=(difference_type n) {
    mIterator += n;
    return *this;
  }

  self_type& operator-=(difference_type n) {
    mIterator -= n;
    return *this;
  }

  bool operator==(const self_type& t) const {
    return (mIterator == t.mIterator);
  }

  bool operator<(const self_type& t) const {
    return (mIterator < t.mIterator);
  }

  const wrapped_iterator& base() const {
    return mIterator;
  }
private:
  wrapped_iterator mIterator;
};

}} // utils::datastructure

#endif /* DATA_STRUCTURE__UTILS_DEREFERENCINGITERATOR_H_ */
