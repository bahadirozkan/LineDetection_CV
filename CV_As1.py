#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Hough transformation
def hough_transform(edges):
    bsize = 1
    r,c = edges.shape
    d = int(np.sqrt((r**2)+(c**2)))
    theta = np.arange(-90, 90, bsize)
    rho = np.arange(-d,d,bsize)

    H = np.zeros((len(rho),len(theta)))

    for y in range(r):
        for x in range(c):
            #if it's an edge point then continue
            if edges[y,x] == 255:

                for t in range(theta.shape[0]):
                    R = int(np.round(x*np.cos(theta[t]*np.pi/180) + y*np.sin(theta[t]*np.pi/180))) + d
                    H[R,t] += 1

    return rho,theta,H

# https://gist.github.com/ilyakava/c2ef8aed4ad510ee3987 
def top_n_rho_theta_pairs(ht_acc_matrix, n, rhos, thetas):

    flat = list(set(np.hstack(ht_acc_matrix)))
    flat_sorted = sorted(flat, key = lambda n: -n)
    coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]
    rho_theta = []
    x_y = []
    for coords_for_val_idx in range(0, len(coords_sorted), 1):
        coords_for_val = coords_sorted[coords_for_val_idx]
        for i in range(0, len(coords_for_val), 1):
            n,m = coords_for_val[i] # n by m matrix
            rho = rhos[n]
            theta = thetas[m]
            rho_theta.append([rho, theta])
            x_y.append([m, n]) # just to unnest and reorder coords_sorted
    return [rho_theta[0:n], x_y]

def valid_point(pt, ymax, xmax):

    x, y = pt
    if x <= xmax and x >= 0 and y <= ymax and y >= 0:
        return True
    else:
        return False

def round_tup(tup):

    x,y = [int(round(num)) for num in tup]
    return (x,y)

def draw_rho_theta_pairs(target_im, pairs):

    im_y_max, im_x_max = np.shape(target_im)
    for i in range(0, len(pairs), 1):
        point = pairs[i]
        rho = point[0]
        theta = point[1] * np.pi / 180 # degrees to radians
        # y = mx + b form
        m = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        # possible intersections on image edges
        left = (0, b)
        right = (im_x_max, im_x_max * m + b)
        top = (-b / m, 0)
        bottom = ((im_y_max - b) / m, im_y_max)

        pts = [pt for pt in [left, right, top, bottom] if valid_point(pt, im_y_max, im_x_max)]
        if len(pts) == 2:
            cv2.line(target_im, round_tup(pts[0]), round_tup(pts[1]), (0,0,255), 1)

def main():
    # Load an color image in grayscale
    img = cv2.imread('im03.png',0)
    #resizing is needed for im01 only but also can be used for others
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    #apply Canny detection
    edges = cv2.Canny(img,40,130)

    cv2.imshow('edges',edges)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
    #write the edge image to file
    cv2.imwrite('edge.jpg', edges)
    print("Edge image saved")

    rhos, thetas, H = hough_transform(edges)

    rho_theta_pairs, x_y_pairs = top_n_rho_theta_pairs(H, 22, rhos, thetas)
    im_w_lines = img.copy()
    draw_rho_theta_pairs(im_w_lines, rho_theta_pairs)

    fig = plt.figure(figsize=(6,6))
    plt.imshow(H, aspect='auto')
    plt.title('Hough Transform Accumulator')
    plt.xlabel('theta', fontsize=10)
    plt.ylabel('rho', fontsize=10)
    fig.savefig("Hough Transform.jpg")
    print("Hough Accumulator saved")

    cv2.imshow("hough lines", im_w_lines)
    # wait to press any key to close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #write the lines image to the disk
    cv2.imwrite('lines.jpg', im_w_lines)
    print("Detected lines saved")

if __name__ == "__main__": main()
