#!/usr/bin/env python3

import numpy as np
import matplotlib.pytplot as plt
import copy


def neighboring_squares(x,y,gridshape):
    """List coordinates of neighboring squares in a grid

    Parameters
    ----------
    x : int
    y : int
        Coordinates of square of interest
    gridshape : tuple
        Tuple of ints, dimensions of grid itself

    Returns
    -------
    list
        list of tuples of (int, int), coordinates of neighboring squares
    """
    out = []
    for xx in range(x-1, x+2):
        if xx >= 0 and xx < gridshape[0]:
            for yy in range(y-1, y+2):
                if yy >=0 and yy <gridshape[1]:
                    if (xx,yy) != (x,y):
                        out.append((xx,yy))
    return(out)


def initiate_grid(grid_x, grid_y, num_mines):
    # Initialize grid of mines
    mines_index = np.random.choice(grid_x*grid_y, num_mines, replace=False)
    grid = np.full((grid_x, grid_y), 0)
    mines_xy = [(int(np.floor(i/grid_x)), i%grid_x) for i in mines_index]
    for (x,y) in mines_xy:
        grid[x][y] = 1
    # Grid of numbers of adjacent mines
    neighbor_mines = np.full((grid_x, grid_y), 0)
    for x in range(grid_x):
        for y in range(grid_y):
            neighbor_mines[x][y] = (sum([grid[xx][yy] for (xx,yy) in neighboring_squares(x,y, (grid_x, grid_y))]))
    for (x,y) in mines_xy:
        neighbor_mines[x][y] = -1
    return(grid, neighbor_mines)


def coords_to_array(coords: list, gridsize: tuple):
    """Convert list of x,y coordinates to array of ones

    Parameters
    ----------
    coords : list
        list of (x,y) tuples representing coordinates of ones
    gridsize: tuple
        dimensions of array
    """
    grid = np.full(gridsize, 0)
    for coord in coords:
        grid[coord[0]][coord[1]] = 1
    return(grid)


class Mines():
    """Representation of minesweeper grid and game history

    
    Attributes
    ----------

    grid : numpy.ndarray
        Array of integers representing positions of mines:
        0 - no mine
        1 - mine
    neighbor_mines : numpy.ndarray
        Array of integers of same size as grid, for each square
        the number of mines in adjoining squares.
        Squares containing mines have value -1
    shape : tuple
        Shape of grid
    mask : numpy.ndarray
        Array of integers of same size as grid, representing
        squares that have been revealed.
        0 - not revealed
        1 - revealed
    tried : list
        Tuples (x,y) of coordinates of squares that have been tried,
        in the order they were tried.
    flagged : numpy.ndarray
        Array of integers of same size as grid, representing squares
        that have been flagged as mines.
        1 - not flagged
        0 - flagged
        Number convention is opposite to that of `grid`, so that the
        correctly flagged mines can be found by simply multiplying
        `grid` with `flagged`.
    mask_snapshot : list
        Snapshots of `mask` arrays at each turn, to capture gameplay
        history.
    """


    def __init__(self, x: int, y:int, num_mines: int, coords=None):
        """Initiate grid with random mine positions

        Parameters
        ----------
        x : int
        y : int
            Size of grid
        num_mines : int
            Number of mines to seed in grid

        TODO: Allow user to define mine positions
        TODO: Snapshot flags and squares tried in addition to masks 
        """
        # Catch attempt to make empty grid
        if num_mines < 1:
            print("Number of mines must be at least 1, setting number to 1")
            num_mines = 1

        self.grid, self.neighbor_mines = initiate_grid(x, y, num_mines)
        self.shape = (x,y)
        self.mask = np.full((x,y), 0)
        self.tried = [] # TODO change this to array, with numbers representing which turn
        self.flagged = np.full((x,y), 1)
        self.mask_snapshot = []


    def flag_square(self, x, y):
        """Flag a square as being possible mine

        Prevents the square from being opened
        """
        if self.mask[x][y] == 0:
            self.flagged[x][y] = 0
        else:
            print(f"Coordinate {str(x)}, {str(y)} already unmasked, ignoring")


    def unflag_square(self, x, y):
        if self.flagged[x][y] == 0:
            self.flagged[x][y] = 1
        else:
            print(f"Coordinate {str(x)}, {str(y)} was not flagged, ignoring")


    def try_square(self, x_try, y_try):
        """Test a square for mines

        Changes attributes `neighbor_mines` and `mask` in-place

        Parameters
        ----------
        x_try : int
        y_try : int
            Coordinates to test

        Returns
        -------
        bool
            True if survived, False if stepped on mine
        """
        if self.flagged[x_try][y_try] != 1:
            print(f"Coordinate {str(x_try)} , {str(y_try)} has already been flagged. Ignoring")
        elif self.mask[x_try][y_try] == 1:
            print(f"Coordinate {str(x_try)} , {str(y_try)} has already been opened. Ignoring")
        else:
            # Record the try
            self.tried.append((x_try, y_try))
            # Reveal the picked square
            self.mask[x_try][y_try] = 1

            # Stepped on a mine
            if self.neighbor_mines[x_try][y_try] < 0:
                # Save result
                self.mask_snapshot.append(copy.deepcopy(self.mask))
                print(f"Coordinate {str(x_try)}, {str(y_try)} had an active mine!")
                return(False)

            # Picked a square with no adjacent mines,
            # reveal all connected squares with zeros
            elif self.neighbor_mines[x_try][y_try] == 0:

                # Algorithm adapted from https://stackoverflow.com/questions/54285527/efficient-algorithm-to-find-connected-components
                squares = [(x_try, y_try)]
                explored = [(x_try, y_try)]
                while len(squares) != 0:
                    current_square = squares[-1]
                    more_to_explore = False
                    for (xx, yy) in neighboring_squares(current_square[0], current_square[1], self.shape):
                        if (xx, yy) not in explored:
                            explored.append((xx,yy))
                            if self.neighbor_mines[xx][yy] == 0:
                                squares.append((xx,yy)) # Push to stack
                                self.mask[xx][yy] = 1 # Unmask
                                more_to_explore = True # Flag more neighbors to explore
                            elif self.neighbor_mines[xx][yy] > 0: # Not a mine
                                self.mask[xx][yy] = 1 # Unmask
                    if not more_to_explore:
                        squares.pop()
                # Re-mask any flagged squares
                self.mask = self.mask * self.flagged
                # Save result
                self.mask_snapshot.append(copy.deepcopy(self.mask))
                return(True)

            # Reveal square that is adjacent to a mine            
            else:
                # Save result
                self.mask_snapshot.append(copy.deepcopy(self.mask))
                return(True)


    ## Graphical output #################################################

    def show_solution(self):
        """Reveal positions of mines"""
        plt.imshow(self.grid)


    def show_revealed(self):
        """Show current state of play

        Mask overlaid on number grid
        """
        plt.imshow(self.mask * self.neighbor_mines)


    def show_tried(self):
        """Show positions of squares that have been tried"""
        plt.imshow(coords_to_array(self.tried, self.shape))


    def show_diagnostic(self):
        """All useful plots together
        
        Returns
        -------
        fig : matplotlib.figure.Figure

        axs : An array of matplotlib.axes.Axes
        """
        fig, axs = plt.subplots(2,3, figsize=(12,8))
        axs[0][0].imshow(self.grid)
        axs[0][0].set_title("Mines")
        axs[0][1].imshow(self.neighbor_mines)
        axs[0][1].set_title("Neighbor mines")
        axs[0][2].imshow(self.mask * self.neighbor_mines)
        axs[0][2].set_title("Revealed")
        axs[1][0].imshow(self.mask)
        axs[1][0].set_title("Mask")
        axs[1][1].imshow(self.flagged)
        axs[1][1].set_title("Flagged")
        axs[1][2].imshow(coords_to_array(self.tried, self.shape))
        axs[1][2].set_title("Tried")
        return(fig, axs)


    def correct_flags(self):
        """Number of mines correctly flagged

        Returns
        -------
        int
        """
        total_flagged = sum(sum(self.flagged == 0))
        total_mines = sum(sum(self.grid == 1))
        correct_flagged = -(self.flagged - 1) * self.grid
        total_correct = sum(sum(correct_flagged))
        return(total_correct)


    def plot_unmask_development(self):
        fig, ax = plt.subplots(figsize=(8,6))
        # Initial zero added manually
        ax.plot(
            list(range(len(self.mask_snapshot) + 1)),
            [0] + [sum(sum(ms)) for ms in self.mask_snapshot])
        return(fig,ax)


    def summary(self):
        """Summarize current state of the minesweeper game

        Returns
        -------
        str
            Text summary of game state
        """
        return(
f"""Grid dimensions: {str(self.shape)}
Total mines: {str(sum(sum(self.grid)))}
Total tries: {str(len(self.tried))}
Squares unmasked: {str(sum(sum(self.mask)))}
Mines flagged: {str(sum(sum(self.flagged == 0)))}
Correct flags: {str(self.correct_flags())}
""")


    ## Autoplay ########################################################

    def autoplay_random(self):
        """Play by trying random squares until we hit a mine"""
        survived = True # False if one hits a mine
        while survived:
            # Pick a random square
            (x_try, y_try) = (np.random.randint(16), np.random.randint(16))
            #if self.mask[x_try][y_try] == 0: # Skip squares already revealed
            if (x_try, y_try) not in self.tried:
                # Check if hit mine, update mask of revealed squares
                survived = self.try_square(x_try, y_try)
            # TODO: break out of endless loop (e.g. grid with no mines)
        print(f"Game over in {str(int(len(self.tried)))} moves")


    def hidden_neighbors(self, x, y):
        """List neighboring squares have not been revealed yet
        
        Returns
        -------
        list
        """
        return(
            [(xx, yy) 
             for (xx, yy) in neighboring_squares(x, y, self.mask.shape)
             if self.mask[xx][yy] == 0])


    def suggest_flags(self, autoflag=False):
        """Flag mine positions using revealed squares

        If number of hidden neighbors equals the number of neighboring mines,
        then all the hidden neighbors must be mines and are hence flagged

        Parameters
        ----------
        autoflag : bool
            Automatically flag the squares. This modifies the Mines object
            in-place.

        Returns
        -------
        list
            tuples (x,y) of putative mined squares to flag
        """
        flagged = []
        # List revealed squares which adjoin mines
        xx, yy = np.where(self.mask*self.neighbor_mines > 0)
        coords = list(zip(xx,yy)) # convert to (x,y) tuples
        for (x,y) in coords:
            hn = self.hidden_neighbors(x,y)
            ns = neighboring_squares(x, y, self.mask.shape)
            if len(hn) == self.neighbor_mines[x][y]:
                flagged.extend(hn)
        # Deduplicate, skip already flagged
        flagged = [(x,y) for (x,y) in list(set(flagged)) if self.flagged[x][y] == 1]
        if autoflag:
            for (x,y) in flagged:
                self.flag_square(x,y)
        return(flagged)


    def suggest_tries(self, autotry=False):
        """Suggest next squares to try, based on flagged squares

        Assumes that mines were correctly flagged.

        Parameters
        ----------
        
        autotry : bool
            Automatically try the squares that are suggested, with
            self.try_squares(). This modifies the Mines object in place.

        Returns
        -------
        list
        """
        # Work backwards from flagged squares to find new squares to reveal
        xx, yy = np.where(self.flagged == 0)
        flagged_coords = list(zip(xx,yy))
        # Initialize list of new squares to open
        to_open_coords = []
        for (flag_x, flag_y) in flagged_coords:
            # Get squares neighboring a flagged mine
            for (n_x, n_y) in neighboring_squares(flag_x, flag_y, self.shape):
                square_value = (self.mask * self.neighbor_mines)[n_x][n_y]
                if square_value > 0:
                    # Get squares neighboring _that_ neighbor square
                    n2_squares = neighboring_squares(n_x, n_y, self.shape)
                    # Number of marked mines adjoining
                    n2_flagged = [coords for coords in n2_squares if coords in flagged_coords]
                    if len(n2_flagged) == square_value:
                        to_open_coords.extend(self.hidden_neighbors(n_x, n_y))
        out = [(x,y) for (x,y) in list(set(to_open_coords)) if (x,y) not in flagged_coords and (x,y) not in self.tried]
        if autotry:
            for (x,y) in out:
                # Skip if already opened by a previous suggested square
                if self.mask[x][y] == 0:
                    # TODO: What happens if try_square() returns False? (i.e. hit a mine)
                    # That should logically never happen...
                    self.try_square(x,y)
        return(out)


    def autoplay_run(self):
        """Autoplay by iteratively flagging mines and opening new squares

        Requires that a first guess has already been made, otherwise it will
        simply terminate because no iterations are possible.

        Terminates when all mines have been found, or when further iterations
        make no progress.
        """
        proceed = True
        n = 0
        while proceed:
            print(f"Iteration {str(n)} ... ")
            ff = self.suggest_flags(autoflag=True)
            tt = self.suggest_tries(autotry=True)
            if self.correct_flags() == sum(sum(self.grid)):
                print("All mines correctly flagged")
                proceed = False
            if len(ff) == 0 and len(tt) == 0:
                print("No further progress possible")
                proceed = False
            n += 1


if __name__ == "__main__":
    print("This is a test")
