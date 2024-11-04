import numpy as np

# Notes:
#
# * A 2D point is represented as a pair of floats.  A 3D point is
#   represented as a triple of floats.
#
# * A 2D line segment is represented as a pair of 2D points. A 3D line
#   segment is represent as a pair of 3D points.
#
# * a wireframe object (i.e., 3D shape) is represented as list of 3D
#   line segments.  A 2D shape is represented as a list of 2D line
#   segments



# this is an example of a wireframe object. A lsit of 3d points. theyd be floats realistically
# example input
test_shape = [((0, 1, 2), (1, 2, 3)),
              ((5, 5, 5), (6, 6, 6)),
              ((-1, -2, -3), (2, 0, 0))]

def shape_to_matrix(shape):
    """converts a wireframe object to a matrix of points in
    homogeneous coordinates

    Parameters:

      shape: list of line segments

    Returns:

      2D numpy array with shape (4, 2 * N) where N = len(shape)

    Example:

      >>> shape_to_matrix(test_shape)
      array([[ 0.,  1.,  5.,  6., -1.,  2.],      x-coordinates of all points
             [ 1.,  2.,  5.,  6., -2.,  0.],      y-coordinates of all points
             [ 2.,  3.,  5.,  6., -3.,  0.],      z-coordinates of all points
             [ 1.,  1.,  1.,  1.,  1.,  1.]])     homogeneous coordinate   

    """

    points = []

    # put them before ebcasue after transpose itll be at the bottomn
    for p1, p2 in shape:
        tuple1 = (p1 + (1,))
        points.append(tuple1)
        tuple2 = (p2 + (1,))
        points.append(tuple2)
    # Convert the list of points to a 2D NumPy array, and then get the transpose of it so it is hoptizontal
    points_matrix = np.array(points).T 
    return points_matrix

def transform_matrix(x_tr, y_tr, z_tr, roll, pitch, yaw):
    """the matrix applied to a shape in order to transformation it by
    translation and rotation

    Parameters:

      x_tr: distance to translate in the x direction (float)
      y_tr: distance to translate in the y direction (float)
      z_tr: distance to translate in the z direction (float)
      roll: angle in radians to rotation about roll axis (float)
      pitch: angle in radians to rotation about pitch axis (float)
      yaw: angle in radians to rotation about yaw axis (float)

    Returns:

      2D numpy array with shape (4, 4)

    Notes:

      * This matrix will be applied to homogeneous coordinates.

      * Make sure that translation is done AFTER rotation.  In
        particular, the centerpoint of the guide axes should remain
        fixed when rotating, even after translation.

    """
    translation_matrix = np.array([
        [1, 0, 0, x_tr],
        [0, 1, 0, y_tr],
        [0, 0, 1, z_tr],
        [0, 0, 0, 1]
    ])

    # roll, changes side-to-side, x axis rotation
    roll_rotation = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])

    pitch_rotation = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])

    yaw_rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # Combine rotation matrices (Order: Roll -> Pitch -> Yaw)
    rotation_matrix = yaw_rotation @ pitch_rotation @ roll_rotation  # Matrix multiplication
    return rotation_matrix @ translation_matrix

test_matrix = np.array(
    [[1, 2, 3, 4],
     [1, 2, 3, 4],
     [1, 2, 3, 4],
     [1, 1, 1, 1]])

def matrix_to_shape(m):
    """converts a set of homogeneous coordinates to a 2D shape (a list
    of 2D line segments)

    Parameters:

      m : 2D matrix with shape (4, 2 * N) where N is the number of
          line segments

    Returns:

      list of 2D line segments (pairs of pairs of floats) after
      projection from distance 10, i.e., with viewing position at the
      point (0, 0, 10)

    Example:

      >>> matrix_to_shape(test_matrix)
      [((1.1111111111111112, 1.1111111111111112), (2.5, 2.5)),
       ((4.285714285714286, 4.285714285714286), (6.666666666666667, 6.666666666666667))]

    Notes:

      * Your values may differ slightly from those in the example
        above, but they should be very similiar.

      * The function numpy.apply_along_axis may be useful here

      * The function zip may be useful here

    """
    # Set the distance from the viewer to the projection plane
    d = 10.0
    
    # Extract the x, y, z coordinates from the matrix
    x3d_arr = m[0, :]  # All x-coordinates, in this case [1, 2, 3, 4]
    y3d_arr = m[1, :]  # All y-coordinates, in this case [1, 2, 3, 4]
    z3d_arr = m[2, :]  # All z-coordinates, in this case [1, 2, 3, 4]


    # Prepare empty list for the projected 2D points
    results = []

    for i in range(0, m.shape[1], 2): # interate with a step of two

      # perspective projection on starting point
      x_start_2d = (x3d_arr[i] / (1 - z3d_arr[i]/d))
      y_start_2d = (y3d_arr[i] / (1 - z3d_arr[i]/d))

      # perspective projection on the end point
      x_end_2d = (x3d_arr[i+1] / (1 - z3d_arr[i+1]/d))
      y_end_2d = (y3d_arr[i+1] / (1 - z3d_arr[i+1]/d))

      starting_pair = (x_start_2d, y_start_2d)
      ending_pair = (x_end_2d, y_end_2d)

      results.append((starting_pair, ending_pair))

  

    return results


def full_transform(x_tr, y_tr, z_tr, roll, pitch, yaw, shape):
    shape_as_matrix = shape_to_matrix(shape)
    my_transformed_matrix = transform_matrix(x_tr, y_tr, z_tr, roll, pitch, yaw)
    final_2d_shape = matrix_to_shape(my_transformed_matrix @ shape_as_matrix)

    return final_2d_shape
    

# For extra credit, create your own wireframe and include an image of
# in the written part of your solution.  For full credit, you must
# make something sufficiently complex, and you must give it a name.
extra_credit_name = "my shape" # TODO

straight_square_edges =[[(-1, 2, -2), (1, 2, -2)],
                        [(2, 1, -2), (2, -1, -2)],
                        [(-1, -2, -2), (1, -2, -2)],
                        [(-2, 1, -2), (-2, -1, -2)],
                        [(-1, 2, 2), (1, 2, 2)],
                        [(2, 1, 2), (2, -1, 2)],
                        [(-1, -2, 2), (1, -2, 2)],
                        [(-2, 1, 2), (-2, -1, 2)],
                      ]


verticals =            [[(1, -2, -2), (-1, -2, 2)],
                        [(-1, -2, -2), (-2, -1, 2)],
                        [(-2, -1, -2), (-2, 1, 2)],
                        [(-2, 1, -2), (-1, 2, 2)],
                        [(-1, 2, -2), (1, 2, 2)],
                        [(1, 2, -2), (2, 1, 2)],
                        [(2, 1, -2), (2, -1, 2)],
                        [(2, -1, -2), (1, -2, 2)],
                      ]



octagon_diagonal_edges = [[(1, 2, -2), (2, 1, -2)],
                          [(2, -1, -2), (1, -2, -2)],
                          [(-1, -2, -2), (-2, -1, -2)], 
                          [(-2, 1, -2), (-1, 2, -2)],

                          [(1, 2, 2), (2, 1, 2)],
                          [(2, -1, 2), (1, -2, 2)],
                          [(-1, -2, 2), (-2, -1, 2)], 
                          [(-2, 1, 2), (-1, 2, 2)],
                  ]

extra_1 =[[(-1, 2, -2.2), (1, 2, -2.2)],
                        [(2, 1, -2.2), (2, -1, -2.2)],
                        [(-1, -2, -2.2), (1, -2, -2.2)],
                        [(-2, 1, -2.2), (-2, -1, -2.2)],
                        [(-1, 2, 2.2), (1, 2, 2.2)],
                        [(2, 1, 2.2), (2, -1, 2.2)],
                        [(-1, -2, 2.2), (1, -2, 2.2)],
                        [(-2, 1, 2.2), (-2, -1, 2.2)],
                      ]

extra_2 = [[(1, 2, -2.2), (2, 1, -2.2)],
                          [(2, -1, -2.2), (1, -2, -2.2)],
                          [(-1, -2, -2.2), (-2, -1, -2.2)], 
                          [(-2, 1, -2.2), (-1, 2, -2.2)],

                          [(1, 2, 2.2), (2, 1, 2.2)],
                          [(2, -1, 2.2), (1, -2, 2.2)],
                          [(-1, -2, 2.2), (-2, -1, 2.2)], 
                          [(-2, 1, 2.2), (-1, 2, 2.2)],
                  ]

straight_square_edges =[[(-1, 2, -2), (1, 2, -2)],
                        [(2, 1, -2), (2, -1, -2)],
                        [(-1, -2, -2), (1, -2, -2)],
                        [(-2, 1, -2), (-2, -1, -2)],
                        [(-1, 2, 2), (1, 2, 2)],

                        [(2, 1, 2), (2, -1, 2)],
                        [(-1, -2, 2), (1, -2, 2)],
                        [(-2, 1, 2), (-2, -1, 2)],
                      ]


extra_3_vertical = [[(-1, 2, -2), (-1, 2, -2.2)],
                    [(1, 2, -2), (1, 2, -2.2)],
                    [(2, 1, -2), (2, 1, -2.2)],
                    [(2, -1, -2), (2, -1, -2.2)],
                    [(-1, -2, -2), (-1, -2, -2.2)],
                    [(1, -2, -2), (1, -2, -2.2)],
                    [(-2, 1, -2), (-2, 1, -2.2)],
                    [(-2, -1, -2), (-2, -1, -2.2)],
                    [(-1, 2, 2), (-1, 2, 2.2)],
                    [(1, 2, 2), (1, 2, 2.2)],
                    [(2, 1, 2), (2, 1, 2.2)],
                    [(2, -1, 2), (2, -1, 2.2)],
                    [(-1, -2, 2), (-1, -2, 2.2)],
                    [(1, -2, 2), (1, -2, 2.2)],
                    [(-2, 1, 2), (-2, 1, 2.2)],
                    [(-2, -1, 2), (-2, -1, 2.2)]
                    ]



                  
    



extra_credit_shape = straight_square_edges + octagon_diagonal_edges + verticals + extra_1 + extra_2 + extra_3_vertical
