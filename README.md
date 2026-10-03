# K-D-Tree-binary-tree-with-LSH-implementation
Two-phase movie search over 946K+ films using a k-d tree for multidimensional filtering and LSH for genre similarity.

**Phase 1 – Multidimensional search (k-d tree)**
A k-d tree is built on 5 numerical attributes:
`budget`, `revenue`, `runtime`, `popularity`, `vote_average`.
It quickly narrows the dataset down to the movies that match the
numerical criteria.

**Phase 2 – Text similarity (LSH)**
Locality-Sensitive Hashing is applied to the `genre_name` attribute
of the Phase 1 results, returning the movies with the most similar genres.
