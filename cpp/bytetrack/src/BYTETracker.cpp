#include "BYTETracker.h"
#include "logger_manager.h"

BYTETracker::BYTETracker(int frame_rate, int track_buffer) {
    track_thresh_ = 0.5;
    high_thresh_  = 0.6;
    match_thresh_ = 0.8;

    frame_id_      = 0;
    max_time_lost_ = int(frame_rate / 30.0 * track_buffer);
    APP_INFO("Init ByteTrack!");
}

BYTETracker::~BYTETracker() {}

std::vector<STrack> BYTETracker::update(const std::vector<Object> & objects) {
    ////////////////// Step 1: Get detections //////////////////
    this->frame_id_++;
    std::vector<STrack> activated_stracks;
    std::vector<STrack> refind_stracks;
    std::vector<STrack> removed_stracks;
    std::vector<STrack> lost_stracks;
    std::vector<STrack> detections;
    std::vector<STrack> detections_low;

    std::vector<STrack> detections_cp;
    std::vector<STrack> tracked_stracks_swap;
    std::vector<STrack> resa, resb;
    std::vector<STrack> output_stracks;

    std::vector<STrack *> unconfirmed;
    std::vector<STrack *> tracked_stracks;
    std::vector<STrack *> strack_pool;
    std::vector<STrack *> r_tracked_stracks;

    if (objects.size() > 0) {
        for (int i = 0; i < objects.size(); i++) {
            std::vector<float> tlbr_;
            tlbr_.resize(4);
            tlbr_[0] = objects[i].rect.x;
            tlbr_[1] = objects[i].rect.y;
            tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
            tlbr_[3] = objects[i].rect.y + objects[i].rect.height;

            float score = objects[i].prob;

            STrack strack(STrack::tlbrToTlwh(tlbr_), score, objects[i].label, objects[i].distance);
            if (score >= track_thresh_) {
                detections.push_back(strack);
            } else {
                detections_low.push_back(strack);
            }
        }
    }

    // Add newly detected tracklets to tracked_stracks
    for (int i = 0; i < this->tracked_stracks_.size(); i++) {
        if (!this->tracked_stracks_[i].is_activated_) {
            unconfirmed.push_back(&this->tracked_stracks_[i]);
        } else {
            tracked_stracks.push_back(&this->tracked_stracks_[i]);
        }
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    strack_pool = jointStracks(tracked_stracks, this->lost_stracks_);
    STrack::multiPredict(strack_pool, this->kalman_filter_);

    std::vector<std::vector<float>> dists;
    int                             dist_size = 0, dist_size_size = 0;
    dists = iouDistance(strack_pool, detections, dist_size, dist_size_size);

    std::vector<std::vector<int>> matches;
    std::vector<int>              u_track, u_detection;
    linearAssignment(dists, dist_size, dist_size_size, match_thresh_, matches, u_track,
                     u_detection);

    for (int i = 0; i < matches.size(); i++) {
        STrack * track = strack_pool[matches[i][0]];
        STrack * det   = &detections[matches[i][1]];
        if (track->state_ == TrackState::TRACKED) {
            track->update(*det, this->frame_id_);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id_, false);
            refind_stracks.push_back(*track);
        }
    }

    ////////////////// Step 3: Second association, using low score dets
    /////////////////////
    for (int i = 0; i < u_detection.size(); i++) {
        detections_cp.push_back(detections[u_detection[i]]);
    }
    detections.clear();
    detections.assign(detections_low.begin(), detections_low.end());

    for (int i = 0; i < u_track.size(); i++) {
        if (strack_pool[u_track[i]]->state_ == TrackState::TRACKED) {
            r_tracked_stracks.push_back(strack_pool[u_track[i]]);
        }
    }

    dists.clear();
    dists = iouDistance(r_tracked_stracks, detections, dist_size, dist_size_size);

    matches.clear();
    u_track.clear();
    u_detection.clear();
    linearAssignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

    for (int i = 0; i < matches.size(); i++) {
        STrack * track = r_tracked_stracks[matches[i][0]];
        STrack * det   = &detections[matches[i][1]];
        if (track->state_ == TrackState::TRACKED) {
            track->update(*det, this->frame_id_);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id_, false);
            refind_stracks.push_back(*track);
        }
    }

    for (int i = 0; i < u_track.size(); i++) {
        STrack * track = r_tracked_stracks[u_track[i]];
        if (track->state_ != TrackState::LOST) {
            track->markLost();
            lost_stracks.push_back(*track);
        }
    }

    // Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections.clear();
    detections.assign(detections_cp.begin(), detections_cp.end());

    dists.clear();
    dists = iouDistance(unconfirmed, detections, dist_size, dist_size_size);

    matches.clear();
    std::vector<int> u_unconfirmed;
    u_detection.clear();
    linearAssignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

    for (int i = 0; i < matches.size(); i++) {
        unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id_);
        activated_stracks.push_back(*unconfirmed[matches[i][0]]);
    }

    for (int i = 0; i < u_unconfirmed.size(); i++) {
        STrack * track = unconfirmed[u_unconfirmed[i]];
        track->markRemoved();
        removed_stracks.push_back(*track);
    }

    ////////////////// Step 4: Init new stracks //////////////////
    for (int i = 0; i < u_detection.size(); i++) {
        STrack * track = &detections[u_detection[i]];
        if (track->score_ < this->high_thresh_) {
            continue;
        }
        track->activate(this->kalman_filter_, this->frame_id_);
        activated_stracks.push_back(*track);
    }

    ////////////////// Step 5: Update state //////////////////
    for (int i = 0; i < this->lost_stracks_.size(); i++) {
        if (this->frame_id_ - this->lost_stracks_[i].endFrame() > this->max_time_lost_) {
            this->lost_stracks_[i].markRemoved();
            removed_stracks.push_back(this->lost_stracks_[i]);
        }
    }

    for (int i = 0; i < this->tracked_stracks_.size(); i++) {
        if (this->tracked_stracks_[i].state_ == TrackState::TRACKED) {
            tracked_stracks_swap.push_back(this->tracked_stracks_[i]);
        }
    }
    this->tracked_stracks_.clear();
    this->tracked_stracks_.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

    this->tracked_stracks_ = jointStracks(this->tracked_stracks_, activated_stracks);
    this->tracked_stracks_ = jointStracks(this->tracked_stracks_, refind_stracks);

    // std::cout << activated_stracks.size() << std::endl;

    this->lost_stracks_ = subStracks(this->lost_stracks_, this->tracked_stracks_);
    for (int i = 0; i < lost_stracks.size(); i++) {
        this->lost_stracks_.push_back(lost_stracks[i]);
    }

    this->lost_stracks_ = subStracks(this->lost_stracks_, this->removed_stracks_);
    for (int i = 0; i < removed_stracks.size(); i++) {
        this->removed_stracks_.push_back(removed_stracks[i]);
    }

    removeDuplicateStracks(resa, resb, this->tracked_stracks_, this->lost_stracks_);

    this->tracked_stracks_.clear();
    this->tracked_stracks_.assign(resa.begin(), resa.end());
    this->lost_stracks_.clear();
    this->lost_stracks_.assign(resb.begin(), resb.end());

    for (int i = 0; i < this->tracked_stracks_.size(); i++) {
        if (this->tracked_stracks_[i].is_activated_) {
            output_stracks.push_back(this->tracked_stracks_[i]);
        }
    }
    return output_stracks;
}
