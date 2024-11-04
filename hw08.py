import numpy as np

# Notes:
# - A 2D point is a pair of floats (x, y).
# - A 3D point is a triple of floats (x, y, z).
# - A 2D line segment is represented as two 2D points.
# - A 3D line segment is represented as two 3D points.
# - A wireframe object (3D shape) is a list of 3D line segments.
# - A 2D shape is a list of 2D line segments.

# Example input representing a wireframe object as a list of 3D line segments
test_shape = [((0, 1, 2), (1, 2, 3)),
              ((5, 5, 5), (6, 6, 6)),
              ((-1, -2, -3), (2, 0, 0))]

def shape_to_matrix(shape):
    """Converts a wireframe object to a matrix of points in homogeneous coordinates.

    Parameters:
      shape: List of 3D line segments represented as pairs of points.

    Returns:
      2D numpy array with shape (4, 2 * N), where N is the number of line segments.

    Example:
      >>> shape_to_matrix(test_shape)
      array([[ 0.,  1.,  5.,  6., -1.,  2.],      # x-coordinates
             [ 1.,  2.,  5.,  6., -2.,  0.],      # y-coordinates
             [ 2.,  3.,  5.,  6., -3.,  0.],      # z-coordinates
             [ 1.,  1.,  1.,  1.,  1.,  1.]])     # homogeneous coordinate
    """
    points = []

    # Construct tuples of 3D points in homogeneous coordinates
    for p1, p2 in shape:
        points.append(p1 + (1,))  # Add homogeneous coordinate (1) for the first point
        points.append(p2 + (1,))  # Add homogeneous coordinate (1) for the second point
    
    # Convert the list of points to a 2D NumPy array and transpose it
    points_matrix = np.array(points).T 
    return points_matrix

def transform_matrix(x_tr, y_tr, z_tr, roll, pitch, yaw):
    """Creates a transformation matrix for translation and rotation.

    Parameters:
      x_tr: Translation distance in the x direction (float).
      y_tr: Translation distance in the y direction (float).
      z_tr: Translation distance in the z direction (float).
      roll: Rotation angle about the x-axis (float).
      pitch: Rotation angle about the y-axis (float).
      yaw: Rotation angle about the z-axis (float).

    Returns:
      2D numpy array representing the transformation matrix with shape (4, 4).

    Notes:
      - Ensure translation is applied after rotation to keep the centerpoint fixed during rotation.
    """
    # Translation matrix for shifting points
    translation_matrix = np.array([
        [1, 0, 0, x_tr],
        [0, 1, 0, y_tr],
        [0, 0, 1, z_tr],
        [0, 0, 0, 1]
    ])

    # Roll rotation (rotation about x-axis)
    roll_rotation = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])

    # Pitch rotation (rotation about y-axis)
    pitch_rotation = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])

    # Yaw rotation (rotation about z-axis)
    yaw_rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Combine the rotation matrices (order: Roll -> Pitch -> Yaw)
    rotation_matrix = yaw_rotation @ pitch_rotation @ roll_rotation  
    return rotation_matrix @ translation_matrix  # Return the combined transformation matrix

# Test matrix for conversion demonstration
test_matrix = np.array(
    [[1, 2, 3, 4],
     [1, 2, 3, 4],
     [1, 2, 3, 4],
     [1, 1, 1, 1]])  # Homogeneous coordinates for testing

def matrix_to_shape(m):
    """Converts a set of homogeneous coordinates to a list of 2D line segments.

    Parameters:
      m: 2D matrix with shape (4, 2 * N) where N is the number of line segments.

    Returns:
      List of 2D line segments (pairs of pairs of floats) after projection from a viewing position at (0, 0, 10).

    Example:
      >>> matrix_to_shape(test_matrix)
      [((1.1111111111111112, 1.1111111111111112), (2.5, 2.5)),
       ((4.285714285714286, 4.285714285714286), (6.666666666666667, 6.666666666666667))]
    """
    # Set the distance from the viewer to the projection plane
    d = 10.0
    
    # Extract x, y, z coordinates from the matrix
    x3d_arr = m[0, :]  # All x-coordinates
    y3d_arr = m[1, :]  # All y-coordinates
    z3d_arr = m[2, :]  # All z-coordinates

    results = []  # Prepare a list to store the projected 2D points

    # Iterate through the matrix in pairs to create line segments
    for i in range(0, m.shape[1], 2): 
        # Perspective projection for starting point
        x_start_2d = x3d_arr[i] / (1 - z3d_arr[i]/d)
        y_start_2d = y3d_arr[i] / (1 - z3d_arr[i]/d)

        # Perspective projection for the end point
        x_end_2d = x3d_arr[i + 1] / (1 - z3d_arr[i + 1]/d)
        y_end_2d = y3d_arr[i + 1] / (1 - z3d_arr[i + 1]/d)

        # Store the 2D points as a tuple of tuples
        results.append(((x_start_2d, y_start_2d), (x_end_2d, y_end_2d)))

    return results  # Return the list of 2D line segments

def full_transform(x_tr, y_tr, z_tr, roll, pitch, yaw, shape):
    """Applies a full transformation (translation and rotation) to a 3D shape and converts it to 2D.

    Parameters:
      x_tr: Translation distance in the x direction.
      y_tr: Translation distance in the y direction.
      z_tr: Translation distance in the z direction.
      roll: Angle of rotation about the x-axis.
      pitch: Angle of rotation about the y-axis.
      yaw: Angle of rotation about the z-axis.
      shape: List of 3D line segments representing the shape.

    Returns:
      List of 2D line segments after transformation and projection.
    """
    shape_as_matrix = shape_to_matrix(shape)  # Convert shape to matrix
    transformation_matrix = transform_matrix(x_tr, y_tr, z_tr, roll, pitch, yaw)  # Get transformation matrix
    final_2d_shape = matrix_to_shape(transformation_matrix @ shape_as_matrix)  # Apply transformation and project

    return final_2d_shape  # Return the final 2D shape

custom_shape_name = "helimix bottle"  # Placeholder for custom shape name
    

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

custom_shape_name = straight_square_edges + octagon_diagonal_edges + verticals + extra_1 + extra_2 + extra_3_vertical
