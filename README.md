# ZOE-to-ILP-Transformer
Python-based solver that transforms Zero-One Equations (ZOE) into Integer Linear Programming (ILP) problems, applying optimization and brute-force techniques to find feasible solutions efficiently. Includes sample input/output and complete documentation.


## Overview
This project provides a Python-based approach to transform and solve **Zero-One Equations (ZOE)** using **Integer Linear Programming (ILP)**. The transformation leverages brute-force and group-wise optimization to efficiently search for solutions even in constraint-heavy environments.

## Files Included
- `Group_8_Project.py`: Main script to convert ZOE to ILP and solve it.
- `DSA_Report.pdf`: Technical documentation including algorithm explanation, examples, and results.
- `sample_input.txt`: Output for a solvable ZOE example.
- `output_file.txt`: Output file showing both solvable and unsolvable ZOEs.

## ▶Usage

### 1. Input Format
Prepare a CSV file where each row represents coefficients of a single zero-one equation. For a system with `n` variables, each row should have `n` values.

### 2. Run the Program
python3 Group_8_Project.py sample_input.txt

