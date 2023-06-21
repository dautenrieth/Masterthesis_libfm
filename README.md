# libFM and OGB Framework

This repository contains a framework that facilitates the loading of graphs from the [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/) and training them using the [libFM](https://github.com/srendle/libfm) library.

A faster and more efficent way of training models solely within python can be found in the [myFM based Repository](https://github.com/dautenrieth/Masterthesis_myfm) or in the [fastFM based Repository](https://github.com/dautenrieth/Masterthesis_fastfm)

## Installation

There are two ways to install the required dependencies:

- Recreate the environment using the provided `.yml` file and [Anaconda](https://www.anaconda.com/download). This is the recommended way to ensure you have all the necessary packages. Get the necessary exe-files from the [libFM](https://github.com/srendle/libfm) repository. This should include libfm.exe, convert.exe and transpose.exe
- Or manually install the [OGB](https://ogb.stanford.edu/docs/home/) and [libFM](https://github.com/srendle/libfm) libraries. Guides on how to do this can be found on the respective sites.

## Usage

Follow these steps to utilize this framework:

1. Modify the `config.ini` file:
   - Choose the desired graph.
   - Activate the desired data parts.
   - Choose the libfm commands you want to execute and put them in the commands.txt file.

2. Run `main.py`.

3. Data will be created automatically (Filestructure should be ensured)

All run data will be automatically saved in an Excel file for convenient examination and comparison.

## Implemented Data Parts

The framework currently supports the following data parts:

- Vectors:
  - Node Embeddings
  - NodeIDs
  - Neighborhood (Binary)
  - Common Neighborhood (Binary)
- Values:
  - Total Common Neighbors
  - Adamic Adar
  - Resource Allocation

Enjoy training your models!