assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers
    
    # Choose all boxes that have length more than 2 in their value
    box_two = [box for box in values.keys() if len(values[box]) == 2]
    
    # Go through each of them and check for their peers in the list of boxes with length greater than 2
    # If any peer match is found check if their box values are the same. If Yes, the naked twins are identified
    common = []
    for i in range(len(box_two)):
        check = box_two[i]
        peer = peers[check]
        for j in range(i+1,len(box_two)):
            if box_two[j] in peer:
                if values[check] == values[box_two[j]]:
                    common.append([check,box_two[j]])
    
    # Make a common list of the peers of the Naked Twins
    # Take each individual digit in the naked twins values and remove them from the value of their common peers
    A = {}
    for i in range(len(common)):
        p1 = peers[common[i][0]]
        p2 = peers[common[i][1]]
        A[i] = [element for element in p1 if element in p2]
        val = values[common[i][0]]
        for v in val:
            for element in A[i]:
                values[element] = values[element].replace(v,'')
                
#                 if len(values[element]) > :
#                     values[element] = values[element].replace(v,'')
#     print('Attempt 1')
#     display(values)
    
    # Check if there are other naked twins formed. If yes re-iterate, If no return the values 
    box_try = [box for box in values.keys() if len(values[box]) == 2]
    if box_try != box_two:
        values = naked_twins(values)
        return values
    else:
        return values
    
def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]
    #pass
    
# Make a list of boxes, row units, column units and square units
boxes = cross(rows,cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
# Find the Diagonal Units
d1 = [[rows[i]+cols[i] for i in range(len(rows))]]
d2 = [[rows[i]+cols[-(i+1)] for i in range(len(rows))]]
# Add them to the unitlist 
unitlist = row_units + column_units + square_units + d1 + d2

#if sudoku_diagonal == True:
    #unitlist = row_units + column_units + square_units + d1 + d2
#else:
    #unitlist = row_units + column_units + square_units 

# Make a list of their peers. Copied over from Udacity Tutorials    
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)
  
def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    # Convert grid to list and iterate over its index. If '.' is found change the index value to '123456789
    # Then convert it to the dictionary format of box_id and Value
    grid = list(grid)
    for i in range(len(grid)):
        if grid[i] == '.':
            grid[i] = '123456789'
    return dict(zip(boxes,grid))
    #pass

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    # Copied over from Udacity Tutorial
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    # As done in Udacity Tutorials
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values

def only_choice(values):
    # As done in Udacity Tutorials
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values

def reduce_puzzle(values):
    # As done in Udacity Tutorials
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    # As done in Udacity Tutorials
    values = reduce_puzzle(values)
    if values == False:
        return False
    if all(len(values[s]) == 1 for s in boxes):
        return values
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    #n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    # Almost glossed this over. THis is the end function that ties all the other ones together
    # Conver grid to dictionary. Go through all steps - Eliminate, Only_Choice, Naked Twins and Search
    values = grid_values(grid)
    answer = search(values)
    return answer

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
