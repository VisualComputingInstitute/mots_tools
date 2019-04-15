import pycocotools.mask as rletools
import sys
from mots_common.io import load_seqmap, load_sequences
from mots_eval.MOTS_metrics import compute_MOTS_metrics


IGNORE_CLASS = 10


def mask_iou(a, b, criterion="union"):
  is_crowd = criterion != "union"
  return rletools.iou([a.mask], [b.mask], [is_crowd])[0][0]


def evaluate_class(gt, results, max_frames, class_id):
  _, results_obj = compute_MOTS_metrics(gt, results, max_frames, class_id, IGNORE_CLASS, mask_iou)
  return results_obj


def run_eval(results_folder, gt_folder, seqmap_filename):
  seqmap, max_frames = load_seqmap(seqmap_filename)
  print("Loading ground truth...")
  gt = load_sequences(gt_folder, seqmap)
  print("Loading results...")
  results = load_sequences(results_folder, seqmap)
  print("Compute KITTI tracking eval with simplified matching and MOTSA")
  print("Evaluate class: Cars")
  results_cars = evaluate_class(gt, results, max_frames, 1)
  print("Evaluate class: Pedestrians")
  results_ped = evaluate_class(gt, results, max_frames, 2)



if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("Usage: python eval.py results_folder gt_folder seqmap")
    sys.exit(1)

  results_folder = sys.argv[1]
  gt_folder = sys.argv[2]
  seqmap_filename = sys.argv[3]

  run_eval(results_folder, gt_folder, seqmap_filename)
