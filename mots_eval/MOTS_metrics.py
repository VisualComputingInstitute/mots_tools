import math
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from collections import defaultdict
import pycocotools.mask as rletools
from mots_common.io import SegmentedObject


class MOTSResults:
  def __init__(self):
    self.n_gt_trajectories = 0
    self.n_tr_trajectories = 0
    self.total_num_frames = 0

    # Evaluation metrics
    self.n_gt = 0  # number of ground truth detections
    self.n_tr = 0  # number of tracker detections minus ignored tracker detections
    self.n_itr = 0  # number of ignored tracker detections
    self.tp = 0  # number of true positives
    self.fp = 0  # number of false positives
    self.fn = 0  # number of false negatives
    self.MOTSA = 0
    self.sMOTSA = 0
    self.MOTSP = 0
    self.MOTSAL = 0
    self.MODSA = 0
    self.MODSP = 0
    self.recall = 0
    self.precision = 0
    self.F1 = 0
    self.FAR = 0
    self.total_cost = 0
    self.fragments = 0
    self.id_switches = 0
    self.MT = 0
    self.PT = 0
    self.ML = 0
    self.IDF1 = 0
    self.IDTP = 0
    self.id_n_tr = 0


# go through all frames and associate ground truth and tracker results
def compute_MOTS_metrics_per_sequence(seq_name, gt_seq, results_seq, max_frames, class_id,
                                      ignore_class, overlap_function):
  results_obj = MOTSResults()
  results_obj.total_num_frames = max_frames + 1
  seq_trajectories = defaultdict(list)

  # To count number of track ids
  gt_track_ids = set()
  tr_track_ids = set()

  # Statistics over the current sequence
  seqtp = 0
  seqfn = 0
  seqfp = 0
  seqitr = 0

  n_gts = 0
  n_trs = 0

  frame_to_ignore_region = {}

  # Iterate over frames in this sequence
  for f in range(max_frames + 1):
    g = []
    dc = []
    t = []

    if f in gt_seq:
      for obj in gt_seq[f]:
        if obj.class_id == ignore_class:
          dc.append(obj)
        elif obj.class_id == class_id:
          g.append(obj)
          gt_track_ids.add(obj.track_id)
    if f in results_seq:
      for obj in results_seq[f]:
        if obj.class_id == class_id:
          t.append(obj)
          tr_track_ids.add(obj.track_id)

    # Handle ignore regions as one large ignore region
    dc = SegmentedObject(mask=rletools.merge([d.mask for d in dc], intersect=False),
                         class_id=ignore_class, track_id=ignore_class)
    frame_to_ignore_region[f] = dc

    tracks_valid = [False for _ in range(len(t))]

    # counting total number of ground truth and tracker objects
    results_obj.n_gt += len(g)
    results_obj.n_tr += len(t)

    n_gts += len(g)
    n_trs += len(t)

    # tmp variables for sanity checks and MODSP computation
    tmptp = 0
    tmpfp = 0
    tmpfn = 0
    tmpc = 0  # this will sum up the overlaps for all true positives
    tmpcs = [0] * len(g)  # this will save the overlaps for all true positives
    # the reason is that some true positives might be ignored
    # later such that the corrsponding overlaps can
    # be subtracted from tmpc for MODSP computation

    # To associate, simply take for each ground truth the (unique!) detection with IoU>0.5 if it exists

    # all ground truth trajectories are initially not associated
    # extend groundtruth trajectories lists (merge lists)
    for gg in g:
      seq_trajectories[gg.track_id].append(-1)
    num_associations = 0
    for row, gg in enumerate(g):
      for col, tt in enumerate(t):
        c = overlap_function(gg, tt)
        if c > 0.5:
          tracks_valid[col] = True
          results_obj.total_cost += c
          tmpc += c
          tmpcs[row] = c
          seq_trajectories[g[row].track_id][-1] = t[col].track_id

          # true positives are only valid associations
          results_obj.tp += 1
          tmptp += 1

          num_associations += 1

    # associate tracker and DontCare areas
    # ignore tracker in neighboring classes
    nignoredtracker = 0  # number of ignored tracker detections

    for i, tt in enumerate(t):
      overlap = overlap_function(tt, dc, "a")
      if overlap > 0.5 and not tracks_valid[i]:
        nignoredtracker += 1

    # count the number of ignored tracker objects
    results_obj.n_itr += nignoredtracker

    # false negatives = non-associated gt instances
    #
    tmpfn += len(g) - num_associations
    results_obj.fn += len(g) - num_associations

    # false positives = tracker instances - associated tracker instances
    # mismatches (mme_t)
    tmpfp += len(t) - tmptp - nignoredtracker
    results_obj.fp += len(t) - tmptp - nignoredtracker
    # tmpfp   = len(t) - tmptp - nignoredtp # == len(t) - (tp - ignoredtp) - ignoredtp
    # self.fp += len(t) - tmptp - nignoredtp

    # update sequence data
    seqtp += tmptp
    seqfp += tmpfp
    seqfn += tmpfn
    seqitr += nignoredtracker

    # sanity checks
    # - the number of true positives minues ignored true positives
    #   should be greater or equal to 0
    # - the number of false negatives should be greater or equal to 0
    # - the number of false positives needs to be greater or equal to 0
    #   otherwise ignored detections might be counted double
    # - the number of counted true positives (plus ignored ones)
    #   and the number of counted false negatives (plus ignored ones)
    #   should match the total number of ground truth objects
    # - the number of counted true positives (plus ignored ones)
    #   and the number of counted false positives
    #   plus the number of ignored tracker detections should
    #   match the total number of tracker detections
    if tmptp < 0:
      print(tmptp)
      raise NameError("Something went wrong! TP is negative")
    if tmpfn < 0:
      print(tmpfn, len(g), num_associations)
      raise NameError("Something went wrong! FN is negative")
    if tmpfp < 0:
      print(tmpfp, len(t), tmptp, nignoredtracker)
      raise NameError("Something went wrong! FP is negative")
    if tmptp + tmpfn != len(g):
      print("seqname", seq_name)
      print("frame ", f)
      print("TP    ", tmptp)
      print("FN    ", tmpfn)
      print("FP    ", tmpfp)
      print("nGT   ", len(g))
      print("nAss  ", num_associations)
      raise NameError("Something went wrong! nGroundtruth is not TP+FN")
    if tmptp + tmpfp + nignoredtracker != len(t):
      print(seq_name, f, len(t), tmptp, tmpfp)
      print(num_associations)
      raise NameError("Something went wrong! nTracker is not TP+FP")

    # compute MODSP
    MODSP_f = 1
    if tmptp != 0:
      MODSP_f = tmpc / float(tmptp)
    results_obj.MODSP += MODSP_f

  assert len(seq_trajectories) == len(gt_track_ids)
  results_obj.n_gt_trajectories = len(gt_track_ids)
  results_obj.n_tr_trajectories = len(tr_track_ids)

  # compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
  if len(seq_trajectories) != 0:
    for g in seq_trajectories.values():
      # all frames of this gt trajectory are not assigned to any detections
      if all([this == -1 for this in g]):
        results_obj.ML += 1
        continue
      # compute tracked frames in trajectory
      last_id = g[0]
      # first detection (necessary to be in gt_trajectories) is always tracked
      tracked = 1 if g[0] >= 0 else 0
      for f in range(1, len(g)):
        if last_id != g[f] and last_id != -1 and g[f] != -1:
          results_obj.id_switches += 1
        if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
          results_obj.fragments += 1
        if g[f] != -1:
          tracked += 1
          last_id = g[f]
      # handle last frame; tracked state is handled in for loop (g[f]!=-1)
      if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1:
        results_obj.fragments += 1

      # compute MT/PT/ML
      tracking_ratio = tracked / float(len(g))
      if tracking_ratio > 0.8:
        results_obj.MT += 1
      elif tracking_ratio < 0.2:
        results_obj.ML += 1
      else:  # 0.2 <= tracking_ratio <= 0.8
        results_obj.PT += 1

    # compute IDF1
    idf1, idtp, id_n_tr = compute_idf1_and_idtp_for_sequence(gt_seq, results_seq, gt_track_ids, tr_track_ids,
                                                             frame_to_ignore_region)
    results_obj.IDF1 = idf1
    results_obj.IDTP = idtp
    #results_obj.id_ign = id_ign
    results_obj.id_n_tr = id_n_tr
  return results_obj


def compute_MOTS_metrics(gt, results, max_frames, class_id, ignore_class, overlap_function):
  """
      Like KITTI tracking eval but with simplified association (when we assume non overlapping masks)
  """
  results_per_seq = {}
  for seq in gt.keys():
    results_seq = {}
    if seq in results:
      results_seq = results[seq]
    results_per_seq[seq] = compute_MOTS_metrics_per_sequence(seq, gt[seq], results_seq, max_frames[seq], class_id,
                                                             ignore_class, overlap_function)

  # Sum up results for all sequences
  results_for_all_seqs = MOTSResults()
  mots_results_attributes = [a for a in dir(results_for_all_seqs) if not a.startswith('__')]
  for attr in mots_results_attributes:
    results_for_all_seqs.__dict__[attr] = sum(obj.__dict__[attr] for obj in results_per_seq.values())

  # Compute aggregate metrics
  for res in results_per_seq.values():
    compute_prec_rec_clearmot(res)
  compute_prec_rec_clearmot(results_for_all_seqs)

  print_summary(list(gt.keys()), results_per_seq, results_for_all_seqs)

  return results_per_seq, results_for_all_seqs


def compute_prec_rec_clearmot(results_obj):
  # precision/recall etc.
  if (results_obj.fp + results_obj.tp) == 0 or (results_obj.tp + results_obj.fn) == 0:
    results_obj.recall = 0.
    results_obj.precision = 0.
  else:
    results_obj.recall = results_obj.tp / float(results_obj.tp + results_obj.fn)
    results_obj.precision = results_obj.tp / float(results_obj.fp + results_obj.tp)
  if (results_obj.recall + results_obj.precision) == 0:
    results_obj.F1 = 0.
  else:
    results_obj.F1 = 2. * (results_obj.precision * results_obj.recall) / (results_obj.precision + results_obj.recall)
  if results_obj.total_num_frames == 0:
    results_obj.FAR = "n/a"
  else:
    results_obj.FAR = results_obj.fp / float(results_obj.total_num_frames)
  # compute CLEARMOT
  if results_obj.n_gt == 0:
    results_obj.MOTSA = -float("inf")
    results_obj.MODSA = -float("inf")
    results_obj.sMOTSA = -float("inf")
  else:
    results_obj.MOTSA = 1 - (results_obj.fn + results_obj.fp + results_obj.id_switches) / float(results_obj.n_gt)
    results_obj.MODSA = 1 - (results_obj.fn + results_obj.fp) / float(results_obj.n_gt)
    results_obj.sMOTSA = (results_obj.total_cost - results_obj.fp - results_obj.id_switches) / float(results_obj.n_gt)
  if results_obj.tp == 0:
    results_obj.MOTSP = float("inf")
  else:
    results_obj.MOTSP = results_obj.total_cost / float(results_obj.tp)
  if results_obj.n_gt != 0:
    if results_obj.id_switches == 0:
      results_obj.MOTSAL = 1 - (results_obj.fn + results_obj.fp + results_obj.id_switches) / float(results_obj.n_gt)
    else:
      results_obj.MOTSAL = 1 - (results_obj.fn + results_obj.fp + math.log10(results_obj.id_switches)) / float(
        results_obj.n_gt)
  else:
    results_obj.MOTSAL = -float("inf")

  if results_obj.total_num_frames == 0:
    results_obj.MODSP = "n/a"
  else:
    results_obj.MODSP = results_obj.MODSP / float(results_obj.total_num_frames)

  if results_obj.n_gt_trajectories == 0:
    results_obj.MT = 0.
    results_obj.PT = 0.
    results_obj.ML = 0.
  else:
    results_obj.MT /= float(results_obj.n_gt_trajectories)
    results_obj.PT /= float(results_obj.n_gt_trajectories)
    results_obj.ML /= float(results_obj.n_gt_trajectories)

  # IDF1
  if results_obj.n_gt_trajectories == 0:
    results_obj.IDF1 = 0.
  else:
    results_obj.IDF1 = (2 * results_obj.IDTP) / (results_obj.n_gt + results_obj.id_n_tr)
  return results_obj


def print_summary(seq_names, results_per_seq, results_for_all_seqs, column_width=14):
  metrics = [("sMOTSA", "sMOTSA"), ("MOTSA", "MOTSA"),
             ("MOTSP", "MOTSP"), ("MOTSAL", "MOTSAL"), ("MODSA", "MODSA"), ("MODSP", "MODSP"),
             ("IDF1", "IDF1"),
             ("Recall", "recall"), ("Prec", "precision"), ("F1", "F1"), ("FAR", "FAR"),
             ("MT", "MT"), ("PT", "PT"), ("ML", "ML"),
             ("TP", "tp"), ("FP", "fp"), ("FN", "fn"),
             ("IDS", "id_switches"), ("Frag", "fragments"),
             ("GT Obj", "n_gt"), ("GT Trk", "n_gt_trajectories"),
             ("TR Obj", "n_tr"), ("TR Trk", "n_tr_trajectories"), ("Ig TR Tck", "n_itr")]
  metrics_names = [tup[0] for tup in metrics]
  metrics_keys = [tup[1] for tup in metrics]
  row_format = "{:>4}" + "".join([("{:>" + str(max(len(name), 4)+2) + "}") for name in metrics_names])
  print(row_format.format("", *metrics_names))

  def format_results_entries(results_obj):
    res = []
    for key in metrics_keys:
      entry = results_obj.__dict__[key]
      if isinstance(entry, float):
        res.append("%.1f" % (entry * 100.0))
      else:
        res.append(str(entry))
    return res

  all_results = format_results_entries(results_for_all_seqs)
  print(row_format.format("all", *all_results))
  for seq in seq_names:
    all_results = format_results_entries(results_per_seq[seq])
    print(row_format.format(seq, *all_results))


def create_summary_KITTI_style(results_obj):
  summary = ""

  summary += "tracking evaluation summary".center(80, "=") + "\n"
  summary += print_entry("Multiple Object Tracking Segmentation Accuracy (sMOTSA)", results_obj.sMOTSA) + "\n"
  summary += print_entry("Multiple Object Tracking Accuracy (MOTSA)", results_obj.MOTSA) + "\n"
  summary += print_entry("Multiple Object Tracking Precision (MOTSP)", results_obj.MOTSP) + "\n"
  summary += print_entry("Multiple Object Tracking Accuracy (MOTSAL)", results_obj.MOTSAL) + "\n"
  summary += print_entry("Multiple Object Detection Accuracy (MODSA)", results_obj.MODSA) + "\n"
  summary += print_entry("Multiple Object Detection Precision (MODSP)", results_obj.MODSP) + "\n"
  summary += "\n"
  summary += print_entry("Recall", results_obj.recall) + "\n"
  summary += print_entry("Precision", results_obj.precision) + "\n"
  summary += print_entry("F1", results_obj.F1) + "\n"
  summary += print_entry("False Alarm Rate", results_obj.FAR) + "\n"
  summary += "\n"
  summary += print_entry("Mostly Tracked", results_obj.MT) + "\n"
  summary += print_entry("Partly Tracked", results_obj.PT) + "\n"
  summary += print_entry("Mostly Lost", results_obj.ML) + "\n"
  summary += "\n"
  summary += print_entry("True Positives", results_obj.tp) + "\n"
  summary += print_entry("False Positives", results_obj.fp) + "\n"
  summary += print_entry("False Negatives", results_obj.fn) + "\n"
  summary += print_entry("Missed Targets", results_obj.fn) + "\n"
  summary += print_entry("ID-switches", results_obj.id_switches) + "\n"
  summary += print_entry("Fragmentations", results_obj.fragments) + "\n"
  summary += "\n"
  summary += print_entry("Ground Truth Objects (Total)", results_obj.n_gt) + "\n"
  summary += print_entry("Ground Truth Trajectories", results_obj.n_gt_trajectories) + "\n"
  summary += "\n"
  summary += print_entry("Tracker Objects (Total)", results_obj.n_tr) + "\n"
  summary += print_entry("Ignored Tracker Objects", results_obj.n_itr) + "\n"
  summary += print_entry("Tracker Trajectories", results_obj.n_tr_trajectories) + "\n"
  summary += "=" * 80

  return summary


def print_entry(key, val, width=(70, 10)):
  s_out = key.ljust(width[0])
  if type(val) == int:
    s = "%%%dd" % width[1]
    s_out += s % val
  elif type(val) == float:
    s = "%%%df" % (width[1])
    s_out += s % val
  else:
    s_out += ("%s" % val).rjust(width[1])
  return s_out



### IDF1 stuff
### code below adapted from https://github.com/shenh10/mot_evaluation/blob/5dd51e5cb7b45992774ea150e4386aa0b02b586f/utils/measurements.py
def compute_idf1_and_idtp_for_sequence(frame_to_gt, frame_to_pred, gt_ids, st_ids, frame_to_ignore_region):
  frame_to_can_be_ignored = {}
  for t in frame_to_pred.keys():
    preds_t = frame_to_pred[t]
    pred_masks_t = [p.mask for p in preds_t]
    ignore_region_t = frame_to_ignore_region[t].mask
    overlap = np.squeeze(rletools.iou(pred_masks_t, [ignore_region_t], [1]), axis=1)
    frame_to_can_be_ignored[t] = overlap > 0.5

  gt_ids = sorted(gt_ids)
  st_ids = sorted(st_ids)
  groundtruth = [[] for _ in gt_ids]
  prediction = [[] for _ in st_ids]
  for t, gts_t in frame_to_gt.items():
    for gt_t in gts_t:
      if gt_t.track_id in gt_ids:
        groundtruth[gt_ids.index(gt_t.track_id)].append((t, gt_t))
  for t in frame_to_pred.keys():
    preds_t = frame_to_pred[t]
    can_be_ignored_t = frame_to_can_be_ignored[t]
    assert len(preds_t) == len(can_be_ignored_t)
    for pred_t, ign_t in zip(preds_t, can_be_ignored_t):
      if pred_t.track_id in st_ids:
        prediction[st_ids.index(pred_t.track_id)].append((t, pred_t, ign_t))
  for gt in groundtruth:
    gt.sort(key=lambda x: x[0])
  for pred in prediction:
    pred.sort(key=lambda x: x[0])

  n_gt = len(gt_ids)
  n_st = len(st_ids)
  cost = np.zeros((n_gt + n_st, n_st + n_gt), dtype=float)
  cost[n_gt:, :n_st] = sys.maxsize  # float('inf')
  cost[:n_gt, n_st:] = sys.maxsize  # float('inf')

  fp = np.zeros(cost.shape)
  fn = np.zeros(cost.shape)
  ign = np.zeros(cost.shape)
  # cost matrix of all trajectory pairs
  cost_block, fp_block, fn_block, ign_block = cost_between_gt_pred(groundtruth, prediction)
  cost[:n_gt, :n_st] = cost_block
  fp[:n_gt, :n_st] = fp_block
  fn[:n_gt, :n_st] = fn_block
  ign[:n_gt, :n_st] = ign_block

  # computed trajectory match no groundtruth trajectory, FP
  for i in range(n_st):
    #cost[i + n_gt, i] = prediction[i].shape[0]
    #fp[i + n_gt, i] = prediction[i].shape[0]
    # don't count fp in case of ignore region
    fps = sum([~x[2] for x in prediction[i]])
    ig = sum([x[2] for x in prediction[i]])
    cost[i + n_gt, i] = fps
    fp[i + n_gt, i] = fps
    ign[i + n_gt, i] = ig

  # groundtruth trajectory match no computed trajectory, FN
  for i in range(n_gt):
    #cost[i, i + n_st] = groundtruth[i].shape[0]
    #fn[i, i + n_st] = groundtruth[i].shape[0]
    cost[i, i + n_st] = len(groundtruth[i])
    fn[i, i + n_st] = len(groundtruth[i])
  # TODO: add error handling here?
  matched_indices = linear_assignment(cost)
  #nbox_gt = sum([groundtruth[i].shape[0] for i in range(n_gt)])
  #nbox_st = sum([prediction[i].shape[0] for i in range(n_st)])
  nbox_gt = sum([len(groundtruth[i]) for i in range(n_gt)])

  nbox_st = sum([len(prediction[i]) for i in range(n_st)])

  #IDFP = 0
  IDFN = 0
  id_ign = 0
  for matched in zip(*matched_indices):
    #IDFP += fp[matched[0], matched[1]]
    IDFN += fn[matched[0], matched[1]]
    # exclude detections which are not matched and ignored from total count
    id_ign += ign[matched[0], matched[1]]
  id_n_tr = nbox_st - id_ign

  IDTP = nbox_gt - IDFN
  #assert IDTP == nbox_st - IDFP
  #IDP = IDTP / (IDTP + IDFP) * 100  # IDP = IDTP / (IDTP + IDFP)
  #IDR = IDTP / (IDTP + IDFN) * 100  # IDR = IDTP / (IDTP + IDFN)
  # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
  IDF1 = 2 * IDTP / (nbox_gt + id_n_tr)
  return IDF1, IDTP, id_n_tr


def cost_between_gt_pred(groundtruth, prediction):
  n_gt = len(groundtruth)
  n_st = len(prediction)
  cost = np.zeros((n_gt, n_st), dtype=float)
  fp = np.zeros((n_gt, n_st), dtype=float)
  fn = np.zeros((n_gt, n_st), dtype=float)
  ign = np.zeros((n_gt, n_st), dtype=float)
  for i in range(n_gt):
    for j in range(n_st):
      fp[i, j], fn[i, j], ign[i, j] = cost_between_trajectories(groundtruth[i], prediction[j])
      cost[i, j] = fp[i, j] + fn[i, j]
  return cost, fp, fn, ign


def cost_between_trajectories(traj1, traj2):
  #[npoints1, dim1] = traj1.shape
  #[npoints2, dim2] = traj2.shape
  npoints1 = len(traj1)
  npoints2 = len(traj2)
  # find start and end frame of each trajectories
  #start1 = traj1[0, 0]
  #end1 = traj1[-1, 0]
  #start2 = traj2[0, 0]
  #end2 = traj2[-1, 0]
  times1 = [x[0] for x in traj1]
  times2 = [x[0] for x in traj2]
  start1 = min(times1)
  start2 = min(times2)
  end1 = max(times1)
  end2 = max(times2)

  ign = [traj2[i][2] for i in range(npoints2)]

  # check frame overlap
  #has_overlap = max(start1, start2) < min(end1, end2)
  # careful, changed this to <=, but I think now it's right
  has_overlap = max(start1, start2) <= min(end1, end2)
  if not has_overlap:
    fn = npoints1
    #fp = npoints2
    # disregard detections which can be ignored
    fp = sum([~x for x in ign])
    ig = sum(ign)
    return fp, fn, ig

  # gt trajectory mapping to st, check gt missed
  matched_pos1 = corresponding_frame(times1, npoints1, times2, npoints2)
  # st trajectory mapping to gt, check computed one false alarms
  matched_pos2 = corresponding_frame(times2, npoints2, times1, npoints1)
  overlap1 = compute_overlap(traj1, traj2, matched_pos1)
  overlap2 = compute_overlap(traj2, traj1, matched_pos2)
  # FN
  fn = sum([1 for i in range(npoints1) if overlap1[i] < 0.5])
  # FP
  # don't count false positive in case of ignore region
  unmatched = [overlap2[i] < 0.5 for i in range(npoints2)]
  #fp = sum([1 for i in range(npoints2) if overlap2[i] < 0.5 and not traj2[i][2]])
  fp = sum([1 for i in range(npoints2) if unmatched[i] and not ign[i]])
  ig = sum([1 for i in range(npoints2) if unmatched[i] and ign[i]])
  return fp, fn, ig


def corresponding_frame(traj1, len1, traj2, len2):
  """
  Find the matching position in traj2 regarding to traj1
  Assume both trajectories in ascending frame ID
  """
  p1, p2 = 0, 0
  loc = -1 * np.ones((len1,), dtype=int)
  while p1 < len1 and p2 < len2:
    if traj1[p1] < traj2[p2]:
      loc[p1] = -1
      p1 += 1
    elif traj1[p1] == traj2[p2]:
      loc[p1] = p2
      p1 += 1
      p2 += 1
    else:
      p2 += 1
  return loc


def compute_overlap(traj1, traj2, matched_pos):
  """
  Compute the loss hit in traj2 regarding to traj1
  """
  overlap = np.zeros((len(matched_pos),), dtype=float)
  for i in range(len(matched_pos)):
    if matched_pos[i] == -1:
      continue
    else:
      mask1 = traj1[i][1].mask
      mask2 = traj2[matched_pos[i]][1].mask
      iou = rletools.iou([mask1], [mask2], [False])[0][0]
      overlap[i] = iou
  return overlap
