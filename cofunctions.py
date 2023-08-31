"""
@param self
LESIM Project
UNIVERSITÀ DEGLI STUDI DEL SANNIO Benevento

Authors: Arman Neyestani,
         Francesco Picariello
Title: Detect and measuring the Angles in Order to spinal cord.
 
"""


import numpy as np
import math

# -----------------------------------
def calculate_angle(a, b, c):
    """_summary_
    Measuring the angle between 3 joints.
    
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    
    # Correct the range from [-2*pi, 2*pi] to [-pi, pi]
    radians = (radians + np.pi) % (2 * np.pi) - np.pi

    angle = radians * 180.0 / np.pi
    
    return angle

def find_angle(line1, line2):
    """_summary_
    This function takes as input two tuples of two tuples each, representing the start
    and end points of two lines in 2D space. It calculates the vectors corresponding
    to these lines, then the dot product and magnitudes of these vectors.
    It then calculates cos(theta) and uses the arccos function to find theta, then converts
    this from radians to degrees. Please note that the function above assumes that the lines
    are defined as ((x1, y1), (x2, y2)), where (x1, y1) is the start point and (x2, y2) is
    the end point of the line.

    Args:
        line1 (tuple): ((x1, y1), (x2, y2))
        line2 (tuple): ((x1, y1), (x2, y2))

    Returns:
        Degree: _description_
    """
    
    # calculate vectors
    u = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    v = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    
    # calculate dot product and magnitudes
    dot_product = u[0]*v[0] + u[1]*v[1]
    magnitude_u = math.sqrt(u[0]**2 + u[1]**2)
    magnitude_v = math.sqrt(v[0]**2 + v[1]**2)
    
    # calculate angle (in degrees)
    cos_theta = dot_product / (magnitude_u * magnitude_v)
    theta = math.acos(cos_theta)
    angle = math.degrees(theta)
    
    return angle
        
def spin_line(shoulder_left,shoulder_right, hip_left, hip_right):
    """_summary_
    The function will find the midpoint of two lines.
    

    Args:
        shoulder_left (list): _description_
        shoulder_right (list): _description_
        hip_left (list): _description_
        hip_right (list): _description_

    Returns:
        list [2*2]: _description_
    """
    midpoint_shoulders=[(shoulder_left[0]+shoulder_right[0])/2,(shoulder_left[1]+shoulder_right[1])/2]
    midpoint_hips=[(hip_left[0]+hip_right[0])/2,(hip_left[1]+hip_right[1])/2]
    
    return midpoint_shoulders, midpoint_hips

def find_intersection(line1, line2):
    """_summary_
    The function that will calculate the intersection point of two lines.
    It uses the formulae for calculating the intersection of two lines given their points.
    Lines are represented as pairs of points, and the function will return a tuple representing
    the intersection point (if it exists), or None if the lines are parallel.

    Args:
        line1 (tuple): ((x1, y1), (x2, y2))
        line2 (tuple): ((x1, y1), (x2, y2))

    Returns:
        point (tuple): (x, y)
    """

    # unpack points
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # calculate differences
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    # calculate denominator
    denom = dx1 * dy2 - dy1 * dx2

    # lines are parallel
    if denom == 0:
        return None

    # calculate numerators
    a = y1 * x2 - y2 * x1
    b = y3 * x4 - y4 * x3

    # calculate intersection point
    x = (a * dx2 - dx1 * b) / denom
    y = (a * dy2 - dy1 * b) / denom
    # ----------------------------------------
       
    return (x, y)

def point_on_line(point, line):
    # Unpack the point and line coordinates
    x, y = point
    line_point1, line_point2 = line

    # Points of line
    x1, y1 = line_point1
    x2, y2 = line_point2

    # Check for a vertical line
    if x1 == x2:
        return x == x1

    # Calculate the slope of the line
    slope = (y2 - y1) / (x2 - x1)

    # Calculate the y-intercept of the line
    intercept = y1 - slope * x1

    # Check if the point lies on the line
    return abs(y - (slope * x + intercept)) < 1e-9  # Used tolerance for float comparison
