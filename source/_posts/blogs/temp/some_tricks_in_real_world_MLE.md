## Feature Engineering

### Handling Missing Values

1. Cause of missing
   1. The value itself (e.g. certain groups of people)
   2. Another variable
   3. No reason
2. Handling
   1. Deletion
      - By Row
      - By Column
   2. Imputation

### Scaling

### Discretization

### Categorical Feature Encoding

[IMPT] industrially adopted encoding trick - hashing

- Hash each category
- New incoming category gets hashed to an existing index
- Random collision not too bad
- Significantly resolved "Unknown category" problem
