/**
 * @file DereferencingVector.h
 * @author Lukas Bossard
 * @date 05.01.2010
 *
 */
#ifndef DATA_STRUCTURE__UTILS_DEREFERENCINGVECTOR_H_
#define DATA_STRUCTURE__UTILS_DEREFERENCINGVECTOR_H_

// standard library includes
#include <vector>

// external library includes
#include <boost/type_traits/is_integral.hpp>
#include <boost/static_assert.hpp>

// local includes
#include "dereferencing_iterator.hpp"

namespace utils {
namespace data_structure {

/**
 * Non owning vector of pointers that dereferences the pointers automatically
 * upon access.
 * *NOTE*: Non owning -> user needs to take care about the life span of the
 *         contained objects
 * For an owning container type consider using boost::ptr_vector
 */
template<class T, class P = T*>
class DereferencingVector {
public:
  typedef T                                                    value_type;
  typedef T&                                                   reference;
  typedef const reference                                      const_reference;
  typedef P                                                    ptr_type;

protected:
  typedef std::vector<ptr_type>                                collection_type;

public:
  typedef typename collection_type::size_type                  size_type;
  typedef DereferencingIterator<
      value_type,
      typename collection_type::iterator>                      iterator;
  typedef DereferencingIterator<
      const value_type,
      typename collection_type::const_iterator>                const_iterator;

  DereferencingVector();

  template<typename insert_iterator>
  DereferencingVector(insert_iterator first, insert_iterator last);

  DereferencingVector(const DereferencingVector<T, P>& vector);

  virtual ~DereferencingVector();

  reference operator[](const size_type index);
  const_reference operator[](const size_type index) const;
  const_reference at(size_type n) const;
  reference at(size_type n);

  ptr_type get_raw(size_type index) const;
  void set_raw(size_type index, ptr_type item);

  iterator begin();
  const_iterator begin() const;

  iterator end();
  const_iterator end() const;

  inline size_type size() const;
  void reserve(const size_type n);
  void resize(const size_type n);
  void clear();

  void destroy_all();

  template<typename insert_iterator>
  void insert(insert_iterator first, insert_iterator last);

  void push_back(ptr_type item);
  void pop_back();
  void erase(const size_type index); // just removes from the vector. no explicit delete[]

private:
  collection_type mCollection;
};
////////////////////////////////////////////////////////////////////////////////
// inline / template implementation

template<class T, class P>
DereferencingVector<T, P>::DereferencingVector() {

}

template<class T, class P>
template<typename insert_iterator>
DereferencingVector<T, P>::DereferencingVector(insert_iterator first,
    insert_iterator last) {
  insert(first, last);
}

template<class T, class P>
DereferencingVector<T, P>::DereferencingVector(
    const DereferencingVector<T, P>& vector) {
  // make some room
  const std::size_t size = vector.mCollection.size();
  resize(size);

  // copy
  for (unsigned int i = 0; i < size; ++i) {
    mCollection[i] = vector.mCollection[i];
  }
}

template<class T, class P>
DereferencingVector<T, P>::~DereferencingVector() {

}

template<class T, class P>
inline typename DereferencingVector<T, P>::size_type DereferencingVector<T, P>::size() const {
  return mCollection.size();
}

template<class T, class P>
void DereferencingVector<T, P>::reserve(const size_type n) {
  return mCollection.reserve(n);
}

template<class T, class P>
void DereferencingVector<T, P>::resize(const size_type n) {
  // if smaller: data gets pointers are just forgotten
  // if size is bigger than actual size -> NULL initialization
  return mCollection.resize(n, NULL);
}

template<class T, class P>
void DereferencingVector<T, P>::clear() {
  mCollection.clear();
}

template<class T, class P>
void DereferencingVector<T, P>::destroy_all() {
  // delete/destroy content
  while (!mCollection.empty()) {
    delete mCollection.back();
    mCollection.pop_back();
  }
}

template<class T, class P>
typename DereferencingVector<T, P>::reference DereferencingVector<T, P>::operator[](
    const size_type index) {
  return *mCollection[index];
}

template<class T, class P>
typename DereferencingVector<T, P>::const_reference DereferencingVector<T, P>::operator[](
    const size_type index) const {
  return *mCollection[index];
}
template<class T, class P>
typename DereferencingVector<T, P>::const_reference DereferencingVector<T, P>::at(
    size_type n) const {
  return *mCollection.at(n);
}

template<class T, class P>
typename DereferencingVector<T, P>::reference DereferencingVector<T, P>::at(
    size_type n) {
  return *mCollection.at(n);
}

template<class T, class P>
typename DereferencingVector<T, P>::ptr_type DereferencingVector<T, P>::get_raw(
    size_type index) const {
  return mCollection.at(index);
}

template<class T, class P>
void
DereferencingVector<T, P>::set_raw(size_type index, DereferencingVector<T, P>::ptr_type item) {
  mCollection.at(index) = item;
}

template<class T, class P>
typename DereferencingVector<T, P>::iterator DereferencingVector<T, P>::begin() {
  return iterator(mCollection.begin());
}
template<class T, class P>
typename DereferencingVector<T, P>::const_iterator DereferencingVector<T, P>::begin() const {
  return const_iterator(mCollection.begin());
}

template<class T, class P>
typename DereferencingVector<T, P>::iterator DereferencingVector<T, P>::end() {
  return iterator(mCollection.end());
}

template<class T, class P>
typename DereferencingVector<T, P>::const_iterator DereferencingVector<T, P>::end() const {
  return const_iterator(mCollection.end());
}

template<class T, class P>
template<typename insert_iterator>
void DereferencingVector<T, P>::insert(insert_iterator first,
    insert_iterator last) {
  // make sure, that we don't get an integral type
  BOOST_STATIC_ASSERT(!boost::is_integral<insert_iterator>::value);

  while (first != last) {
    this->push_back(*first);
    ++first;
  }
}

template<class T, class P>
void DereferencingVector<T, P>::push_back(ptr_type item) {
  mCollection.push_back(item);
}

template<class T, class P>
void DereferencingVector<T, P>::pop_back() {
  mCollection.pop_back();
}

template<class T, class P>
void DereferencingVector<T, P>::erase(const size_type index) {
  mCollection.erase(mCollection.begin() + index);
}

}
} // namespace utils::data_structure
#endif /* DATA_STRUCTURE__UTILS_DEREFERENCINGVECTOR_H_ */
