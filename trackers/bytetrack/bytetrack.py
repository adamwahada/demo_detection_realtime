"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box
https://arxiv.org/abs/2110.06864
"""

import numpy as np
from collections import deque
from .kalman_filter import KalmanFilter
from .matching import linear_assignment, iou_distance


class STrack:
    """Single target track with Kalman filter."""
    
    shared_kalman = KalmanFilter()
    track_id_count = 0
    
    def __init__(self, tlwh, score):
        """
        Args:
            tlwh: bounding box in (top, left, width, height) format
            score: detection confidence score
        """
        # Convert tlwh to tlbr (top, left, bottom, right)
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        
        self.track_id = 0
        self.state = 'new'
        
        self.frame_id = 0
        self.start_frame = 0
    
    def predict(self):
        """Predict next state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != 'tracked':
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
    
    @staticmethod
    def multi_predict(stracks):
        """Predict multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != 'tracked':
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
    
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.state = 'tracked'
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
    
    def update(self, new_track, frame_id):
        """Update a matched track."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = 'tracked'
        self.is_activated = True
        
        self.score = new_track.score
    
    @property
    def tlwh(self):
        """Get current position in (top, left, width, height) format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self):
        """Get current position in (top, left, bottom, right) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to (center_x, center_y, aspect_ratio, height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_xyah(self):
        """Convert current position to (center_x, center_y, aspect_ratio, height) format."""
        return self.tlwh_to_xyah(self.tlwh)
    
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert tlbr to tlwh format."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert tlwh to tlbr format."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.frame_id})'
    
    @staticmethod
    def next_id():
        """Get next track ID."""
        STrack.track_id_count += 1
        return STrack.track_id_count


class BYTETracker:
    """
    ByteTrack tracker implementation.
    
    Args:
        track_thresh: Detection confidence threshold for tracking
        track_buffer: Number of frames to keep lost tracks
        match_thresh: IOU threshold for matching
        frame_rate: Video frame rate
    """
    
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30, use_low_conf=False):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.match_thresh = match_thresh
        self.use_low_conf = use_low_conf 
    
    def update(self, output_results, img_info, img_size):
        """
        Update tracker with new detections.
        
        Args:
            output_results: Detection results in format [x1, y1, x2, y2, score]
            img_info: Image info (height, width)
            img_size: Image size (height, width)
            
        Returns:
            List of active tracks in format [x1, y1, x2, y2, track_id]
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        
        # Separate detections by confidence
        remain_inds = scores > self.track_thresh
        inds_low = scores > self.track_thresh * 0.9 
        inds_high = scores < self.track_thresh
        
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        
        # Create detections from high-confidence detections
        detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
        
        # Create detections_second from low-confidence detections (if enabled)
        if self.use_low_conf and len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Step 2: First association, with high score detection boxes
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # Step 3: Second association, with low score detection boxes
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 'tracked']
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == 'lost':
                track.state = 'lost'
                lost_stracks.append(track)
        
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = 'removed'
            removed_stracks.append(track)
        
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = 'removed'
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'tracked']
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # Get output
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        return np.array([[*track.tlbr, track.track_id] for track in output_stracks])


def joint_stracks(tlista, tlistb):
    """Join two track lists."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Subtract tlistb from tlista."""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks based on IOU."""
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
