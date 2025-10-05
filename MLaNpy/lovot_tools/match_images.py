import argparse

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="feature point matches between two images")
    parser.add_argument('imga', help='file path to image a')
    parser.add_argument('imgb', help='file path to image b')
    parser.add_argument('--out', help='file path of output')
    parser.add_argument('--scale-ratio', type=float, default=2.4, help='resize scale ratio')
    parser.add_argument('--ratio-th', type=float, default=0.7,
                        help='threshold of distance ratio to second matches to determine as good.')
    args = parser.parse_args()
    return args


def match_images(img_a, img_b, ratio_th=0.7):
    detector = cv2.BRISK_create()

    gray1 = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio_th * n.distance:
            good.append([m])

    matchesMask = extract_inliers(kp1, kp2, good)

    # cv2.drawMatchesKnn expects list of lists as matches.
    matches_img = cv2.drawMatchesKnn(img_a, kp1, img_b, kp2, good, None,
                                     flags=2,
                                     matchesMask=matchesMask)

    scale = 2
    image_size = (int(matches_img.shape[1] / scale), int(matches_img.shape[0] / scale))
    matches_img = cv2.resize(matches_img, image_size)

    cv2.imshow('match', matches_img)
    cv2.waitKey(0)

    return matches_img


def extract_inliers(kp_a, kp_b, matches):
    src_pts = np.float32([kp_a[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(M)
    matchesMask = mask.tolist()
    return matchesMask


def run():
    args = parse_args()

    img_a = cv2.imread(args.imga)
    image_size = (int(img_a.shape[1] / args.scale_ratio), int(img_a.shape[0] / args.scale_ratio))
    img_a = cv2.resize(img_a, image_size)

    img_b = cv2.imread(args.imgb)
    image_size = (int(img_b.shape[1] / args.scale_ratio), int(img_b.shape[0] / args.scale_ratio))
    img_b = cv2.resize(img_b, image_size)

    output = match_images(img_a, img_b, args.ratio_th)

    if args.out:
        cv2.imwrite(args.out, output)


if __name__ == '__main__':
    run()
