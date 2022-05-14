import open3d
import typing
import os
import argparse

p = argparse.ArgumentParser()
p.add_argument('-g', '--gt', required=True, type=str, default='gt.ply', help='The file path of ground truth.')
p.add_argument('-p', '--pred', required=True, type=str, default='pred.ply', help='The file path of predicted.')

args = p.parse_args()

def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float=0.01) -> typing.Tuple[float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


def main():
    gt = open3d.io.read_point_cloud(args.gt)
    pr = open3d.io.read_point_cloud(args.pred)
    f_score, precision, recall = calculate_fscore(gt, pr)
    print(f"The f-score is {f_score}, precision is {precision} and recall is {recall}.")
    return 0

if __name__=='__main__':
    exit(main())
