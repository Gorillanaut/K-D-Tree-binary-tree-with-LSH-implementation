# k-d Tree + LSH Movie Search

Two-phase search over a dataset of 946,000+ movies.

**Phase 1 – Multidimensional search (k-d tree)**
A k-d tree is built on 5 numerical attributes:
`budget`, `revenue`, `runtime`, `popularity`, `vote_average`.
It quickly narrows the dataset down to the movies that match the
numerical criteria.

**Phase 2 – Text similarity (LSH)**
Locality-Sensitive Hashing is applied to the `genre_name` attribute
of the Phase 1 results, returning the movies with the most similar genres.

## How to Run

### Requirements
- **Visual Studio Code** (or any Python editor): [installation guide](https://code.visualstudio.com/docs/setup/setup-overview)
- **Python 3**: [download](https://www.python.org/downloads/)
- **pip** (usually included with Python): [installation guide](https://pip.pypa.io/en/stable/installation/)

### 1. Download the code
Clone the repository:

    git clone https://github.com/Gorillanaut/K-D-Tree-binary-tree-with-LSH-implementation.git
    cd K-D-Tree-binary-tree-with-LSH-implementation

Or click **Code → Download ZIP** on this page and extract it.

### 2. Download the dataset
The dataset is too large for GitHub, so it is hosted on Google Drive:

**[Download the dataset](https://drive.google.com/file/d/1IffoJ-Q0l1LLVXaXUEtEEme9cDlSL7LY/view?usp=drive_link)**

Place the downloaded file in the project's root folder (the same folder as `kd_lsh.py`).

### 3. Install the libraries

    pip install numpy pandas

### 4. Run the k-d tree search
Open the project folder in VS Code and run `kd_lsh.py`, or from the terminal:

    python kd_lsh.py

The program will:

1. Build the k-d tree and the LSH index, and print the construction time of each.
2. Ask for a lower and an upper bound for each numerical attribute, in this order:
   `budget`, `revenue`, `runtime`, `popularity`, `vote_average`.
3. Ask for a genre text, which LSH compares against the movies' genres.
4. Print the matching movies in the terminal (if any are found).
