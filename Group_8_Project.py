# Import Libraries
import numpy as np
import itertools
import pandas as pd
import warnings
import argparse
import sys
import csv


# Output File
output_file_path = './output_file.txt_new'

# Function to validate the input data
def validate_input_from_csv(input_file_path):
    
    # List of errors encountered during validation
    errors_list = []  
    
    with open(input_file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        row_lengths = None
        for row_number, row in enumerate(reader, start=1):
            
            # Check row length consistency
            if row_lengths is None:
                row_lengths = len(row)
            elif len(row) != row_lengths:
                errors_list.append(f"Row {row_number} has fewer variables than the expected input data.")
                
            # Check if the last column contains 1 as an integer
            if int(row[-1]) != 1:
                errors_list.append(f"Last column value must be '1' at row {row_number}.")
            
            # Check if the second to last column contains '='
            if '=' not in row[-2]:
                errors_list.append(f"The input must have '=' only in the second-to-last column at row {row_number}.")
                
            # Check all other columns for 0 or 1 as integers
            for column_number, value in enumerate(row[:-2], start=1):
                try:
                    value = int(value)
                    if value != 0 and value != 1:
                        errors_list.append(f"At row {row_number}, column {column_number}, the input should consist of only zeros or ones.")
                except ValueError:
                    errors_list.append(f"The input at row {row_number}, column {column_number} is not an integer and is invalid.")
                
    return errors_list

# Function to read the matrices from the CSV file
def read_matrix_from_csv(input_file_path):
    matrix_a = []
    matrix_b = []
    
    # Store the negation of values for matrix A
    neg_matrix_a = []
    
    # Store the negation of values for vector b   
    neg_matrix_b = [] 
    
    # Store the lengths of rows except for the last two columns
    row_lengths = []  
    with open(input_file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            
            # Calculate the length of the row except for the last two columns
            row_length = len(row) - 2
            row_lengths.append(row_length)
            
            # Convert all elements to integers except the last one
            row_a = []
            
            # Store the negation of values for the current row in matrix A
            neg_row_a = []  
            
            for value in row[:-2]:
                value_int = int(value)
                row_a.append(value_int)
                    
                # Add negation of the value to the neg_row_a list
                neg_row_a.append(-value_int)  
    
            matrix_a.append(row_a)
            
            # Add neg_row_a to the neg_matrix_a list
            neg_matrix_a.append(neg_row_a)  
            
            # Convert the last element to an integer and store it in a separate list
            last_value = int(row[-1])
            matrix_b.append([last_value])
            
            # Add negation of the value to the neg_matrix_b list
            neg_matrix_b.append([-last_value])  

    # Adding identity matrix to matrix_a
    max_row_length = max(row_lengths)
    identity_matrix = np.eye(max_row_length, dtype=int)
    matrix_a.extend(identity_matrix.tolist())
    var_constr_b = [0 if i%2!=0 else 1 for i in range(2*max_row_length)]
    # Adding negation of identity matrix to neg_matrix_a
    neg_identity_matrix = -identity_matrix
    neg_matrix_a.extend(neg_identity_matrix.tolist())
    
    
    return matrix_a, matrix_b, neg_matrix_a, neg_matrix_b, var_constr_b, row_lengths

def write_matrices_to_txt(matrix_a, matrix_b, neg_matrix_a, neg_matrix_b, var_constr_b, txt_file_path):
    
        print("Matrix A:\n")
        for i in range(len(matrix_a)):
            print(' '.join(map(str, matrix_a[i])))
            print(' '.join(map(str, neg_matrix_a[i])))
        
        print("\nVector b:\n")
        for i in range(len(matrix_b)):
            print(str(matrix_b[i][0]))
            print(str(neg_matrix_b[i][0]))
        for i in range(len(var_constr_b)):
            print(str(var_constr_b[i]))
            

# Function to merge common groups
def merge_lists(lists):
    # Initialize variables
    merged = True
    while merged:
        merged = False
        result = []
        while lists:
            first, *rest = lists
            lists = rest
            for other in rest:
                if set(first) & set(other):
                    first += other
                    lists.remove(other)
                    merged = True
            # Add merged list, removing duplicates
            result.append(list(set(first)))
            
        # Prepare for next iteration if any merge happened
        lists = result 
    return lists

# Function to convert the zero-one equtions to linear integer programing
def ZOE_to_ILP(constraints, original_df, indices):
    constraint_arr = np.array(constraints[0][0][0])
    num_vars = constraint_arr.shape[0]
    
    best_solution = None
    
    # Generate all possible combinations of binary values for variables
    counter = 0
    for combination in itertools.product([0, 1], repeat=num_vars):
        counter+=1
        valid = True
        for constraint in constraints:
            lhs = sum(constraint[0][0][i] * combination[i] for i in range(num_vars))
            if constraint[1] == "=" and lhs != constraint[2]:
                valid = False
                break
        if valid:
            best_solution = combination
            return best_solution
        
    if not valid:
        return None
    for i in range(len(indices)):
        if indices[i] == 0:
            best_solution = best_solution[0:i] + [0] + best_solution[i:]

    return best_solution


# Function to print the constraint equation
def zoe(zoe_df, original_df, indices):
    
    # Create a copy of the DataFrame
    zoe_df_copy = zoe_df.copy()

# Iterate over columns and drop those with all zeros
    for col in zoe_df_copy.columns:
        if not np.any(zoe_df_copy[col]):
            zoe_df_copy.drop(col, axis=1, inplace=True)
    

    num_vars = zoe_df.shape[1] - 2
    num_constraints = zoe_df.shape[0]
   
    # Initialize lists to store coefficients and constraints
    constraints = []
    # Input constraints    
    for i in range(num_constraints):
        constraint = []
        constraint.append(list(zoe_df.iloc[i,:-2]))
        relation = zoe_df.iloc[i,-2]
        rhs = int(zoe_df.iloc[i,-1])
        constraints.append((constraint, relation, rhs))
    zero_one_equation = ""

    for i in range(num_constraints):
        constraint_eq = ""
        for j in range(num_vars):
            if constraints[i][0][0][j] > 0:
                constraint_eq += f"({constraints[i][0][0][j]} * x{j+1}) + "
            elif constraints[i][0][0][j] < 0:
                constraint_eq += f"({-constraints[i][0][0][j]} * (1 - x{j+1})) + "
        constraint_eq = constraint_eq[:-3]
        if constraints[i][1] == "<=":
            zero_one_equation += f"{constraint_eq} <= {constraints[i][2]} \n"
        elif constraints[i][1] == ">=":
            zero_one_equation += f"{constraint_eq} >= {constraints[i][2]} \n "
        elif constraints[i][1] == "=":
            zero_one_equation += f"{constraint_eq} == {constraints[i][2]} \n "
    
    print("\nZero-one equations:\n")
    print(f"{zero_one_equation[:-3]}")

    best_solution = ZOE_to_ILP(constraints, original_df, indices)
    print("\nSolution for this group:", best_solution)
    return best_solution

# Main function
def main(input_file_path):
    
    # Validate the input CSV file
    validation_errors = validate_input_from_csv(input_file_path)
    if validation_errors:
        print("Errors found in the CSV file:")
        for error in validation_errors:
            print(error)
    else:
        # Redirect standard output to the output file
        sys.stdout = open(output_file_path, 'w')
        
        # Read matrices from CSV file
        matrix_a, matrix_b, neg_matrix_a, neg_matrix_b, var_constr_b, row_lengths = read_matrix_from_csv(input_file_path)
    
        # Write matrices to text file
        write_matrices_to_txt(matrix_a, matrix_b, neg_matrix_a, neg_matrix_b, var_constr_b, output_file_path)
    
        num_vars = row_lengths[0]
        zoe_df = pd.read_csv(input_file_path,names=[i for i in range(num_vars+2)])
        num_constraints = zoe_df.shape[0]

        # Initialize lists to store coefficients and constraints
        constraints = []
        for i in range(num_constraints):
            constraint = []
            constraint.append(list(zoe_df.iloc[i,:-2]))
            relation = zoe_df.iloc[i,-2]
            rhs = int(zoe_df.iloc[i,-1])
            constraints.append((constraint, relation, rhs)) 
            
            
        # Convert constraints to a NumPy array
        A = np.array([i[0][0] for i in constraints])
        
        # Convert the constriants to possible groups
        possible_groups = []
        for i in range(A.shape[1]):
            if np.any(A[:,i]):
                list1 = np.squeeze(np.where(A[:,i] == 1)).tolist()
                if type(list1) != type([]):
                    list1 = [list1]
                possible_groups.append(list1)
        
        # Merge all the possible groups
        possible_groups = merge_lists(possible_groups)
        
        # Iterate each group and find the possible solutions
        sol_arr = []  
        for group in possible_groups:
            indices = np.zeros(zoe_df.iloc[group,:-2].shape[1])
            for row in group:
                indices = np.logical_or(indices, np.array(zoe_df.iloc[row,:-2]))
            solution = zoe(zoe_df.iloc[group], zoe_df, indices)
            if solution is None:
                print("For the given ZOE, no solution exists")
                break
            else:
                sol_arr.append(solution)
        if not solution==None:
            final_sol = np.zeros(zoe_df.iloc[group,:-2].shape[1])
            for sol in sol_arr:
                final_sol = np.logical_or(final_sol, sol)
            final_sol = final_sol.astype(int)
            print("\nOverall solution: ")
            for i, xi in enumerate(final_sol):
                print(f"x_{i+1} = {xi}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input file path")
    parser.add_argument("input_file_path", help="Path to the input CSV file")
    args = parser.parse_args()
    main(args.input_file_path)