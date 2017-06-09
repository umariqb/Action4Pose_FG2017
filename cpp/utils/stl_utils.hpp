/*
 * stl_utils.hpp
 *
 *  Created on: Feb 8, 2013
 *      Author: lbossard
 */

#ifndef UTILS__STL_UTILS_HPP_
#define UTILS__STL_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <tr1/unordered_set>
//------------------------------------------------------------------------------
namespace { // hidden
struct PairFirstRetriever {
  template<typename T>
  typename T::first_type operator()(const T& keyValuePair) const {
    return keyValuePair.first;
  }
};
struct PairSecondRetriever {
  template<typename T>
  typename T::second_type operator()(const T& keyValuePair) const {
    return keyValuePair.second;
  }
};
struct PairRetriever {
  template<typename T>
  T operator()(const T& keyValuePair) const {
    return keyValuePair;
  }
};
}
//------------------------------------------------------------------------------
namespace utils {
namespace stl {

//------------------------------------------------------------------------------
template<class T>
bool is_in(const typename T::key_type& key, const T& container) {
  return (container.find(key) != container.end());
}

//------------------------------------------------------------------------------
/**
 * Fills the map componets retrieved by the retriever into the resultcontainer
 * @param map
 * @param resultContainer
 */
template<typename T_Map, typename T_ResultContainer, typename T_Retriever>
void get_components(const T_Map& container,
    T_ResultContainer& resultContainer) {
  std::transform(
      container.begin(),
      container.end(),
      std::inserter<T_ResultContainer>(resultContainer, resultContainer.end()),
      T_Retriever());
}

template<typename T_Map, typename T_ResultContainer>
void get_keys(const T_Map& map, T_ResultContainer& resultContainer) {
  get_components<T_Map, T_ResultContainer, PairFirstRetriever>(
      map,
      resultContainer);
}

template<typename T_Map, typename T_ResultContainer>
void get_values(const T_Map& map, T_ResultContainer& resultContainer) {
  get_components<T_Map, T_ResultContainer, PairSecondRetriever>(
      map,
      resultContainer);
}

template<typename T_Map, typename T_ResultContainer>
void get_pairs(const T_Map& map, T_ResultContainer& resultContainer) {
  get_components<T_Map, T_ResultContainer, PairRetriever>(
      map,
      resultContainer);
}

//------------------------------------------------------------------------------
template<typename T, typename M>
void vector_to_index(T it, const T end, M& map) {
  std::size_t i = 0;
  while (it != end){
    map[*it] = i;
    ++it;
    ++i;
  }
}

template<typename T, typename M>
void vector_to_index(T vector, M& map) {
  vector_to_index(vector.begin(), vector.end(), map);
}
//------------------------------------------------------------------------------
/**
 * Constructs in place the set of unique elements while retaining the original
 * order
 * @param collection
 */
template<typename T>
void uniquify(std::vector<T>& collection) {

  std::tr1::unordered_set<T> known_elements;
  std::size_t next_unique_idx = 0;
  const std::size_t collection_size = collection.size();
  for (unsigned int i = 0; i < collection_size; ++i) {
    const T& element = collection[i];
    if (known_elements.insert(element).second) {
      collection[next_unique_idx++] = element;
    }
  }
  collection.resize(next_unique_idx);
}

} /* namespace stl */
} /* namespace utils */
#endif /* UTILS__STL_UTILS_HPP_ */
