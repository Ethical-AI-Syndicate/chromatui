pub mod bayesian {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub enum MatchType {
        Exact,
        Prefix,
        WordStart,
        Substring,
        Fuzzy,
    }

    impl MatchType {
        pub fn prior_odds(&self) -> f64 {
            match self {
                MatchType::Exact => 99.0,
                MatchType::Prefix => 9.0,
                MatchType::WordStart => 4.0,
                MatchType::Substring => 2.0,
                MatchType::Fuzzy => 1.0 / 3.0,
            }
        }

        pub fn bf_word_boundary(&self) -> f64 {
            match self {
                MatchType::Exact => 1.0,
                MatchType::Prefix => 1.5,
                MatchType::WordStart => 2.0,
                MatchType::Substring => 1.2,
                MatchType::Fuzzy => 1.0,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ScoringEvidence {
        pub match_type: MatchType,
        pub word_boundary_bonus: f64,
        pub position_factor: f64,
        pub gap_penalty: f64,
        pub tag_match_bonus: f64,
        pub length_factor: f64,
        pub log_posterior: f64,
    }

    pub struct BayesianScorer;

    impl BayesianScorer {
        pub fn new() -> Self {
            Self
        }

        pub fn score(&self, query: &str, target: &str, tags: &[String]) -> (f64, ScoringEvidence) {
            let match_type = self.classify_match(query, target);
            let prior_odds: f64 = match_type.prior_odds();

            let word_boundary_bonus: f64 = if target.starts_with(query)
                || target.split_whitespace().any(|w| w.starts_with(query))
            {
                match_type.bf_word_boundary()
            } else {
                1.0
            };

            let position_factor: f64 =
                1.0 / (target.find(query).map(|i| i + 1).unwrap_or(1) as f64).sqrt();

            let gap_penalty: f64 = self.compute_gap_penalty(query, target);

            let tag_match_bonus: f64 = if tags
                .iter()
                .any(|t| t.to_lowercase().contains(&query.to_lowercase()))
            {
                3.0
            } else {
                1.0
            };

            let length_factor: f64 = 1.0 / (target.len() as f64).sqrt();

            let log_posterior: f64 = prior_odds.ln()
                + word_boundary_bonus.ln()
                + position_factor.ln()
                + gap_penalty.ln()
                + tag_match_bonus.ln()
                + length_factor.ln();

            let posterior: f64 = 1.0 / (1.0 + (-log_posterior).exp());

            (
                posterior,
                ScoringEvidence {
                    match_type,
                    word_boundary_bonus,
                    position_factor,
                    gap_penalty,
                    tag_match_bonus,
                    length_factor,
                    log_posterior,
                },
            )
        }

        fn classify_match(&self, query: &str, target: &str) -> MatchType {
            let q = query.to_lowercase();
            let t = target.to_lowercase();

            if q == t {
                MatchType::Exact
            } else if t.starts_with(&q) {
                MatchType::Prefix
            } else if t.split_whitespace().any(|w| w.starts_with(&q)) {
                MatchType::WordStart
            } else if t.contains(&q) {
                MatchType::Substring
            } else {
                MatchType::Fuzzy
            }
        }

        fn compute_gap_penalty(&self, query: &str, target: &str) -> f64 {
            if query.is_empty() {
                return 1.0;
            }

            let mut gap_count = 0usize;
            let mut prev_idx: Option<usize> = None;
            let target_lower = target.to_lowercase();
            let query_lower = query.to_lowercase();

            for c in query_lower.chars() {
                if let Some(idx) = target_lower.find(c) {
                    if let Some(prev) = prev_idx {
                        if idx > prev + 1 {
                            gap_count += 1;
                        }
                    }
                    prev_idx = Some(idx);
                }
            }

            (gap_count as f64 + 1.0).recip()
        }
    }

    impl Default for BayesianScorer {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_exact_match() {
            let scorer = BayesianScorer::new();
            let (score, evidence) = scorer.score("test", "test", &[]);
            assert_eq!(evidence.match_type, MatchType::Exact);
            assert!(score > 0.9);
        }

        #[test]
        fn test_prefix_match() {
            let scorer = BayesianScorer::new();
            let (score, evidence) = scorer.score("test", "testing", &[]);
            assert_eq!(evidence.match_type, MatchType::Prefix);
            assert!(score > 0.5);
        }

        #[test]
        fn test_tag_bonus() {
            let scorer = BayesianScorer::new();
            let (_, with_tags) = scorer.score("git", "push", &["git".to_string()]);
            let (_, without_tags) = scorer.score("git", "push", &[]);
            assert!(with_tags.tag_match_bonus > without_tags.tag_match_bonus);
        }

        #[test]
        fn test_gap_penalty() {
            let scorer = BayesianScorer::new();
            let (_, continuous) = scorer.score("abc", "abc", &[]);
            let (_, gapped) = scorer.score("abc", "aXbXc", &[]);
            assert!(continuous.gap_penalty >= gapped.gap_penalty);
        }
    }
}

pub mod bocpd {
    use std::collections::VecDeque;

    #[derive(Debug, Clone)]
    pub struct RegimeDetector {
        hazard_rate: f64,
        run_length_posterior: VecDeque<f64>,
        max_runs: usize,
        steady_mean: f64,
        burst_mean: f64,
        burst_posterior: f64,
    }

    impl RegimeDetector {
        pub fn new() -> Self {
            Self {
                hazard_rate: 1.0 / 50.0,
                run_length_posterior: VecDeque::from(vec![1.0]),
                max_runs: 100,
                steady_mean: 200.0,
                burst_mean: 20.0,
                burst_posterior: 0.5,
            }
        }

        pub fn with_params(hazard_rate: f64, steady_mean: f64, burst_mean: f64) -> Self {
            Self {
                hazard_rate,
                run_length_posterior: VecDeque::from(vec![1.0]),
                max_runs: 100,
                steady_mean,
                burst_mean,
                burst_posterior: 0.5,
            }
        }

        pub fn update(&mut self, inter_arrival_ms: f64) -> f64 {
            let steady_rate = 1.0 / self.steady_mean;
            let burst_rate = 1.0 / self.burst_mean;
            let hazard = self.hazard_rate;

            let mut p_changepoint_total = 0.0;
            let mut p_burst_total = 0.0;

            let mut new_posterior = VecDeque::with_capacity(self.max_runs + 1);
            new_posterior.push_back(0.0);

            for (r, &prob) in self.run_length_posterior.iter().enumerate() {
                let p_steady = steady_rate * (-inter_arrival_ms * steady_rate).exp();
                let p_burst = burst_rate * (-inter_arrival_ms * burst_rate).exp();

                let p_continue_steady = prob * (1.0 - hazard) * p_steady;
                let p_continue_burst = prob * (1.0 - hazard) * p_burst;
                let p_cp = prob * hazard;

                p_changepoint_total += p_cp;
                p_burst_total += p_continue_burst;

                if r + 1 >= new_posterior.len() {
                    new_posterior.push_back(0.0);
                }
                new_posterior[r + 1] += p_continue_steady + p_continue_burst;
            }

            new_posterior[0] += p_changepoint_total;

            if new_posterior.is_empty() {
                new_posterior.push_back(1.0);
            }

            let total: f64 = new_posterior.iter().sum();
            if total > 0.0 {
                for p in new_posterior.iter_mut() {
                    *p /= total;
                }
            }

            while new_posterior.len() > self.max_runs {
                new_posterior.pop_back();
            }

            self.run_length_posterior = new_posterior;

            let p_steady_obs = steady_rate * (-inter_arrival_ms * steady_rate).exp();
            let p_burst_obs = burst_rate * (-inter_arrival_ms * burst_rate).exp();
            let obs_total = p_steady_obs + p_burst_obs;
            if obs_total > 0.0 {
                self.burst_posterior = p_burst_obs / obs_total;
            }

            p_burst_total
        }

        pub fn regime(&self) -> Regime {
            if self.burst_posterior > 0.7 {
                Regime::Burst
            } else if self.burst_posterior < 0.3 {
                Regime::Steady
            } else {
                Regime::Transitional
            }
        }
    }

    impl Default for RegimeDetector {
        fn default() -> Self {
            Self::new()
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Regime {
        Steady,
        Burst,
        Transitional,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_steady_regime() {
            let mut detector = RegimeDetector::new();
            for _ in 0..10 {
                detector.update(200.0);
            }
            assert_eq!(detector.regime(), Regime::Steady);
        }

        #[test]
        fn test_burst_regime() {
            let mut detector = RegimeDetector::new();
            for _ in 0..10 {
                detector.update(20.0);
            }
            assert_eq!(detector.regime(), Regime::Burst);
        }
    }
}

pub mod cusum {
    #[derive(Debug, Clone)]
    pub struct CusumDetector {
        pub s: f64,
        pub k: f64,
        pub h: f64,
        pub mu0: f64,
    }

    impl CusumDetector {
        pub fn new() -> Self {
            Self {
                s: 0.0,
                k: 0.5,
                h: 5.0,
                mu0: 0.0,
            }
        }

        pub fn with_params(k: f64, h: f64, mu0: f64) -> Self {
            Self { s: 0.0, k, h, mu0 }
        }

        pub fn update(&mut self, x: f64) -> bool {
            self.s = (self.s + x - self.mu0 - self.k).max(0.0);
            self.s > self.h
        }

        pub fn reset(&mut self) {
            self.s = 0.0;
        }

        pub fn value(&self) -> f64 {
            self.s
        }
    }

    impl Default for CusumDetector {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_cusum_alert() {
            let mut detector = CusumDetector::with_params(0.5, 3.0, 0.0);
            assert!(!detector.update(2.0));
            assert!(!detector.update(2.0));
            assert!(detector.update(2.0));
        }

        #[test]
        fn test_cusum_reset() {
            let mut detector = CusumDetector::with_params(0.5, 3.0, 0.0);
            let _ = detector.update(10.0);
            detector.reset();
            assert_eq!(detector.value(), 0.0);
        }
    }
}

pub mod eprocess {
    use std::collections::VecDeque;

    pub struct EProcess {
        wealth: f64,
        lambda: f64,
        mu0: f64,
        history: VecDeque<f64>,
    }

    impl EProcess {
        pub fn new() -> Self {
            Self {
                wealth: 1.0,
                lambda: 0.5,
                mu0: 0.0,
                history: VecDeque::new(),
            }
        }

        pub fn with_params(lambda: f64, mu0: f64) -> Self {
            Self {
                wealth: 1.0,
                lambda,
                mu0,
                history: VecDeque::new(),
            }
        }

        pub fn update(&mut self, x: f64) -> f64 {
            let increment = self.lambda * (x - self.mu0) - (self.lambda * self.lambda) / 2.0;
            self.wealth *= 1.0 + increment;
            self.history.push_back(increment);
            if self.history.len() > 1000 {
                self.history.pop_front();
            }
            self.wealth
        }

        pub fn wealth(&self) -> f64 {
            self.wealth
        }

        pub fn alert_threshold(&self, alpha: f64) -> bool {
            self.wealth >= 1.0 / alpha
        }

        pub fn reset(&mut self) {
            self.wealth = 1.0;
            self.history.clear();
        }
    }

    impl Default for EProcess {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_wealth_growth() {
            let mut ep = EProcess::with_params(0.5, 0.0);
            ep.update(1.0);
            ep.update(1.0);
            assert!(ep.wealth() > 1.0);
        }

        #[test]
        fn test_alert_threshold() {
            let mut ep = EProcess::new();
            for _ in 0..1000 {
                ep.update(10.0);
            }
            assert!(ep.alert_threshold(0.05));
        }
    }
}

pub mod fenwick {
    pub struct FenwickTree {
        tree: Vec<f64>,
        n: usize,
    }

    impl FenwickTree {
        pub fn new(size: usize) -> Self {
            Self {
                tree: vec![0.0; size + 1],
                n: size,
            }
        }

        pub fn with_values(values: &[f64]) -> Self {
            let n = values.len();
            let mut tree = vec![0.0; n + 1];
            for (i, &v) in values.iter().enumerate() {
                tree[i + 1] = v;
            }
            for i in 1..=n {
                let parent = i + (i & i.wrapping_neg());
                if parent <= n {
                    tree[parent] += tree[i];
                }
            }
            Self { tree, n }
        }

        pub fn update(&mut self, index: usize, delta: f64) {
            let mut idx = index + 1;
            while idx <= self.n {
                self.tree[idx] += delta;
                idx += idx & idx.wrapping_neg();
            }
        }

        pub fn sum(&self, index: usize) -> f64 {
            let mut idx = index + 1;
            let mut result = 0.0;
            while idx > 0 {
                result += self.tree[idx];
                idx -= idx & idx.wrapping_neg();
            }
            result
        }

        pub fn range_sum(&self, left: usize, right: usize) -> f64 {
            if left > right {
                return 0.0;
            }
            self.sum(right) - if left > 0 { self.sum(left - 1) } else { 0.0 }
        }

        pub fn find_prefix(&self, target: f64) -> Option<usize> {
            let mut idx = 0usize;
            let mut bit = 1usize << (self.n.next_power_of_two().trailing_zeros() as usize);

            let mut current_target = target;
            while bit > 0 {
                let next = idx + bit;
                if next <= self.n && self.tree[next] < current_target {
                    idx = next;
                    current_target -= self.tree[next];
                }
                bit >>= 1;
            }

            if idx < self.n {
                Some(idx)
            } else {
                None
            }
        }

        pub fn len(&self) -> usize {
            self.n
        }

        pub fn is_empty(&self) -> bool {
            self.n == 0
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_update_and_sum() {
            let mut ft = FenwickTree::new(5);
            ft.update(0, 1.0);
            ft.update(2, 3.0);
            assert_eq!(ft.sum(0), 1.0);
            assert_eq!(ft.sum(2), 4.0);
        }

        #[test]
        fn test_range_sum() {
            let mut ft = FenwickTree::new(5);
            for i in 0..5 {
                ft.update(i, (i + 1) as f64);
            }
            assert_eq!(ft.range_sum(1, 3), 9.0);
        }

        #[test]
        fn test_find_prefix() {
            let ft = FenwickTree::with_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(ft.find_prefix(3.5), Some(2));
            assert_eq!(ft.find_prefix(6.0), Some(2));
            assert_eq!(ft.find_prefix(20.0), None);
        }
    }
}

pub mod animation {
    use std::time::{Duration, Instant};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Easing {
        Linear,
        EaseIn,
        EaseOut,
        EaseInOut,
    }

    impl Easing {
        pub fn apply(self, t: f64) -> f64 {
            match self {
                Easing::Linear => t,
                Easing::EaseIn => t * t,
                Easing::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
                Easing::EaseInOut => {
                    if t < 0.5 {
                        2.0 * t * t
                    } else {
                        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                    }
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Animation {
        pub start_value: f64,
        pub end_value: f64,
        pub duration: Duration,
        pub easing: Easing,
        pub start_time: Option<Instant>,
    }

    impl Animation {
        pub fn new(start: f64, end: f64, duration: Duration) -> Self {
            Self {
                start_value: start,
                end_value: end,
                duration,
                easing: Easing::EaseInOut,
                start_time: None,
            }
        }

        pub fn with_easing(mut self, easing: Easing) -> Self {
            self.easing = easing;
            self
        }

        pub fn start(&mut self) {
            self.start_time = Some(Instant::now());
        }

        pub fn value(&self) -> f64 {
            if let Some(start) = self.start_time {
                let elapsed = start.elapsed();
                let t =
                    (elapsed.as_millis() as f64 / self.duration.as_millis() as f64).clamp(0.0, 1.0);
                let eased = self.easing.apply(t);
                self.start_value + (self.end_value - self.start_value) * eased
            } else {
                self.start_value
            }
        }

        pub fn is_complete(&self) -> bool {
            if let Some(start) = self.start_time {
                start.elapsed() >= self.duration
            } else {
                false
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct DampedSpring {
        pub position: f64,
        pub velocity: f64,
        pub target: f64,
        pub stiffness: f64,
        pub damping: f64,
    }

    impl DampedSpring {
        pub fn new(target: f64) -> Self {
            Self {
                position: target,
                velocity: 0.0,
                target,
                stiffness: 150.0,
                damping: 10.0,
            }
        }

        pub fn with_params(mut self, stiffness: f64, damping: f64) -> Self {
            self.stiffness = stiffness;
            self.damping = damping;
            self
        }

        pub fn set_target(&mut self, target: f64) {
            self.target = target;
        }

        pub fn update(&mut self, dt: f64) {
            let displacement = self.position - self.target;
            let spring_force = -self.stiffness * displacement;
            let damping_force = -self.damping * self.velocity;
            let acceleration = spring_force + damping_force;

            self.velocity += acceleration * dt;
            self.position += self.velocity * dt;
        }

        pub fn update_stable(&mut self, dt: f64) {
            let max_dt = 1.0 / 120.0;
            if dt <= max_dt {
                self.update(dt);
                return;
            }

            let steps = (dt / max_dt).ceil() as usize;
            let sub_dt = dt / steps as f64;
            for _ in 0..steps {
                self.update(sub_dt);
            }
        }

        pub fn is_settled(&self, threshold: f64) -> bool {
            (self.position - self.target).abs() < threshold && self.velocity.abs() < threshold
        }

        pub fn critical_damping(stiffness: f64) -> f64 {
            2.0 * stiffness.sqrt()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_easing_linear() {
            assert!((Easing::Linear.apply(0.5) - 0.5).abs() < 0.001);
        }

        #[test]
        fn test_easing_ease_in() {
            assert!(Easing::EaseIn.apply(0.5) < 0.5);
        }

        #[test]
        fn test_easing_ease_out() {
            assert!(Easing::EaseOut.apply(0.5) > 0.5);
        }

        #[test]
        fn test_damped_spring_settles() {
            let mut spring = DampedSpring::new(100.0);
            spring.position = 0.0;

            for _ in 0..1000 {
                spring.update(0.016);
                if spring.is_settled(0.01) {
                    break;
                }
            }

            assert!(spring.is_settled(0.01));
        }

        #[test]
        fn test_damped_spring_stable_large_dt() {
            let mut spring = DampedSpring::new(10.0);
            spring.position = 0.0;
            spring.update_stable(0.5);
            assert!(spring.position.is_finite());
            assert!(spring.velocity.is_finite());
        }
    }
}

pub mod diff {
    use std::collections::{HashMap, HashSet};

    pub struct DirtySpanTracker {
        _width: usize,
        dirty_rows: HashSet<usize>,
        row_spans: HashMap<usize, Vec<(usize, usize)>>,
    }

    impl DirtySpanTracker {
        pub fn new(width: usize) -> Self {
            Self {
                _width: width,
                dirty_rows: HashSet::new(),
                row_spans: HashMap::new(),
            }
        }

        pub fn mark_dirty(&mut self, row: usize, col: usize) {
            self.dirty_rows.insert(row);
            self.row_spans.entry(row).or_default().push((col, col + 1));
        }

        pub fn mark_dirty_range(&mut self, row: usize, start_col: usize, end_col: usize) {
            self.dirty_rows.insert(row);
            self.row_spans
                .entry(row)
                .or_default()
                .push((start_col, end_col));
        }

        pub fn get_dirty_rows(&self) -> impl Iterator<Item = &usize> {
            self.dirty_rows.iter()
        }

        pub fn total_dirty_cells(&self) -> usize {
            self.row_spans
                .values()
                .map(|spans| {
                    if spans.is_empty() {
                        return 0;
                    }

                    let mut sorted = spans.clone();
                    sorted.sort_by_key(|(s, _)| *s);

                    let mut total = 0usize;
                    let mut current = sorted[0];
                    for &(s, e) in sorted.iter().skip(1) {
                        if s <= current.1 {
                            current.1 = current.1.max(e);
                        } else {
                            total += current.1 - current.0;
                            current = (s, e);
                        }
                    }
                    total + (current.1 - current.0)
                })
                .sum()
        }

        pub fn merge_spans(&mut self) {
            for spans in self.row_spans.values_mut() {
                spans.sort_by_key(|(s, _)| *s);
                let mut merged = Vec::new();
                let mut current = spans[0];

                for &(s, e) in spans.iter().skip(1) {
                    if s <= current.1 {
                        current.1 = current.1.max(e);
                    } else {
                        merged.push(current);
                        current = (s, e);
                    }
                }
                merged.push(current);
                *spans = merged;
            }
        }
    }

    pub struct SummedAreaTable {
        sat: Vec<Vec<usize>>,
        width: usize,
        height: usize,
    }

    impl SummedAreaTable {
        pub fn new(width: usize, height: usize) -> Self {
            let mut sat = vec![vec![0usize; width + 1]; height + 1];
            for y in 1..=height {
                for x in 1..=width {
                    sat[y][x] = sat[y - 1][x] + sat[y][x - 1] - sat[y - 1][x - 1] + 1;
                }
            }
            Self { sat, width, height }
        }

        pub fn from_dense(dense: &[Vec<bool>]) -> Self {
            let height = dense.len();
            let width = dense.first().map(|r| r.len()).unwrap_or(0);
            let mut sat = vec![vec![0usize; width + 1]; height + 1];

            for y in 1..=height {
                for x in 1..=width {
                    let val = dense[y - 1][x - 1] as usize;
                    sat[y][x] = sat[y - 1][x] + sat[y][x - 1] - sat[y - 1][x - 1] + val;
                }
            }
            Self { sat, width, height }
        }

        pub fn sum(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> usize {
            if x2 > self.width || y2 > self.height {
                return 0;
            }
            self.sat[y2][x2] - self.sat[y1][x2] - self.sat[y2][x1] + self.sat[y1][x1]
        }

        pub fn density(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> f64 {
            let cells = self.sum(x1, y1, x2, y2);
            let total = (x2 - x1) * (y2 - y1);
            if total == 0 {
                0.0
            } else {
                cells as f64 / total as f64
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_dirty_span_tracker() {
            let mut tracker = DirtySpanTracker::new(80);
            tracker.mark_dirty_range(0, 10, 20);
            tracker.mark_dirty_range(0, 15, 25);

            assert!(tracker.dirty_rows.contains(&0));
            assert_eq!(tracker.total_dirty_cells(), 15);
        }

        #[test]
        fn test_sat() {
            let sat = SummedAreaTable::new(3, 3);
            assert_eq!(sat.sum(0, 0, 1, 1), 1);
            assert_eq!(sat.sum(0, 0, 3, 3), 9);
        }

        #[test]
        fn test_sat_from_dense() {
            let dense = vec![
                vec![true, false, true],
                vec![false, false, false],
                vec![true, true, false],
            ];
            let sat = SummedAreaTable::from_dense(&dense);
            assert_eq!(sat.sum(0, 0, 2, 2), 1);
        }
    }
}

pub mod hints {
    #[derive(Debug, Clone)]
    pub struct HintArm {
        pub id: String,
        pub alpha: f64,
        pub beta: f64,
        pub display_cost: f64,
    }

    impl HintArm {
        pub fn new(id: impl Into<String>, display_cost: f64) -> Self {
            Self {
                id: id.into(),
                alpha: 1.0,
                beta: 1.0,
                display_cost,
            }
        }

        pub fn expected_utility(&self) -> f64 {
            self.alpha / (self.alpha + self.beta)
        }

        pub fn variance(&self) -> f64 {
            let s = self.alpha + self.beta;
            (self.alpha * self.beta) / (s * s * (s + 1.0))
        }

        pub fn voi(&self) -> f64 {
            self.variance().sqrt()
        }
    }

    #[derive(Debug, Clone)]
    pub struct HintScore {
        pub id: String,
        pub value: f64,
        pub expected_utility: f64,
        pub voi: f64,
        pub display_cost: f64,
    }

    #[derive(Debug, Clone)]
    pub struct BayesianHintRanker {
        arms: Vec<HintArm>,
        w_voi: f64,
        lambda_cost: f64,
        hysteresis_eps: f64,
        last_top: Option<String>,
    }

    impl BayesianHintRanker {
        pub fn new() -> Self {
            Self {
                arms: Vec::new(),
                w_voi: 0.2,
                lambda_cost: 0.3,
                hysteresis_eps: 0.05,
                last_top: None,
            }
        }

        pub fn with_params(mut self, w_voi: f64, lambda_cost: f64, hysteresis_eps: f64) -> Self {
            self.w_voi = w_voi;
            self.lambda_cost = lambda_cost;
            self.hysteresis_eps = hysteresis_eps;
            self
        }

        pub fn add_hint(&mut self, id: impl Into<String>, display_cost: f64) {
            self.arms.push(HintArm::new(id, display_cost));
        }

        pub fn observe(&mut self, id: &str, useful: bool) {
            if let Some(arm) = self.arms.iter_mut().find(|a| a.id == id) {
                if useful {
                    arm.alpha += 1.0;
                } else {
                    arm.beta += 1.0;
                }
            }
        }

        pub fn rank(&mut self) -> Vec<HintScore> {
            let mut scores: Vec<HintScore> = self
                .arms
                .iter()
                .map(|arm| {
                    let e = arm.expected_utility();
                    let v = arm.voi();
                    let value = e + self.w_voi * v - self.lambda_cost * arm.display_cost;
                    HintScore {
                        id: arm.id.clone(),
                        value,
                        expected_utility: e,
                        voi: v,
                        display_cost: arm.display_cost,
                    }
                })
                .collect();

            scores.sort_by(|a, b| b.value.total_cmp(&a.value));

            if let Some(prev) = self.last_top.clone() {
                if let Some(prev_score) = scores.iter().find(|s| s.id == prev) {
                    if let Some(curr_top) = scores.first() {
                        if curr_top.id != prev
                            && (curr_top.value - prev_score.value) <= self.hysteresis_eps
                        {
                            let idx_prev = scores
                                .iter()
                                .position(|s| s.id == prev)
                                .expect("previous top should exist");
                            let prev_item = scores.remove(idx_prev);
                            scores.insert(0, prev_item);
                        }
                    }
                }
            }

            self.last_top = scores.first().map(|s| s.id.clone());
            scores
        }
    }

    impl Default for BayesianHintRanker {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn useful_hint_ranks_higher() {
            let mut ranker = BayesianHintRanker::new();
            ranker.add_hint("copy", 0.1);
            ranker.add_hint("help", 0.1);
            for _ in 0..20 {
                ranker.observe("copy", true);
                ranker.observe("help", false);
            }
            let ranked = ranker.rank();
            assert_eq!(ranked[0].id, "copy");
        }

        #[test]
        fn hysteresis_prevents_small_swaps() {
            let mut ranker = BayesianHintRanker::new().with_params(0.0, 0.0, 1.0);
            ranker.add_hint("a", 0.0);
            ranker.add_hint("b", 0.0);
            let first = ranker.rank();
            let top = first[0].id.clone();
            ranker.observe("a", true);
            ranker.observe("b", true);
            let second = ranker.rank();
            assert_eq!(second[0].id, top);
        }
    }
}

pub mod capability {
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub struct CapabilityPosterior {
        logit: f64,
        evidence: HashMap<String, f64>,
    }

    impl CapabilityPosterior {
        pub fn new(prior_p: f64) -> Self {
            let p = prior_p.clamp(1e-6, 1.0 - 1e-6);
            Self {
                logit: (p / (1.0 - p)).ln(),
                evidence: HashMap::new(),
            }
        }

        pub fn add_log_bf(&mut self, source: impl Into<String>, log_bf: f64) {
            self.logit += log_bf;
            self.evidence.insert(source.into(), log_bf);
        }

        pub fn probability(&self) -> f64 {
            1.0 / (1.0 + (-self.logit).exp())
        }

        pub fn logit(&self) -> f64 {
            self.logit
        }

        pub fn evidence(&self) -> &HashMap<String, f64> {
            &self.evidence
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn positive_evidence_increases_probability() {
            let mut cap = CapabilityPosterior::new(0.5);
            let p0 = cap.probability();
            cap.add_log_bf("da2", 1.0);
            assert!(cap.probability() > p0);
        }

        #[test]
        fn negative_evidence_decreases_probability() {
            let mut cap = CapabilityPosterior::new(0.8);
            let p0 = cap.probability();
            cap.add_log_bf("env", -2.0);
            assert!(cap.probability() < p0);
        }
    }
}

pub mod voi {
    #[derive(Debug, Clone)]
    pub struct VoiSampler {
        pub alpha: f64,
        pub beta: f64,
        pub value_scale: f64,
        pub sample_cost: f64,
        pub min_interval_ms: u64,
        pub max_interval_ms: u64,
        pub last_sample_ms: Option<u64>,
    }

    impl VoiSampler {
        pub fn new() -> Self {
            Self {
                alpha: 1.0,
                beta: 9.0,
                value_scale: 1.0,
                sample_cost: 0.08,
                min_interval_ms: 100,
                max_interval_ms: 1000,
                last_sample_ms: None,
            }
        }

        pub fn observe(&mut self, violation: bool) {
            if violation {
                self.alpha += 1.0;
            } else {
                self.beta += 1.0;
            }
        }

        pub fn variance(&self) -> f64 {
            let a = self.alpha;
            let b = self.beta;
            (a * b) / (((a + b) * (a + b)) * (a + b + 1.0))
        }

        pub fn expected_variance_after_sample(&self) -> f64 {
            let a = self.alpha;
            let b = self.beta;
            let p = a / (a + b);
            let var_if_success =
                ((a + 1.0) * b) / (((a + b + 1.0) * (a + b + 1.0)) * (a + b + 2.0));
            let var_if_failure =
                (a * (b + 1.0)) / (((a + b + 1.0) * (a + b + 1.0)) * (a + b + 2.0));
            p * var_if_success + (1.0 - p) * var_if_failure
        }

        pub fn voi(&self) -> f64 {
            (self.variance() - self.expected_variance_after_sample()).max(0.0)
        }

        pub fn should_sample(&self, now_ms: u64) -> bool {
            let elapsed = self
                .last_sample_ms
                .map(|t| now_ms.saturating_sub(t))
                .unwrap_or(self.max_interval_ms);

            if elapsed >= self.max_interval_ms {
                return true;
            }
            if elapsed < self.min_interval_ms {
                return false;
            }

            self.voi() * self.value_scale >= self.sample_cost
        }

        pub fn mark_sampled(&mut self, now_ms: u64) {
            self.last_sample_ms = Some(now_ms);
        }
    }

    impl Default for VoiSampler {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn max_interval_forces_sample() {
            let sampler = VoiSampler::new();
            assert!(sampler.should_sample(1000));
        }

        #[test]
        fn min_interval_blocks_sample() {
            let mut sampler = VoiSampler::new();
            sampler.mark_sampled(1000);
            assert!(!sampler.should_sample(1050));
        }
    }
}

pub mod conformal {
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub struct ConformalAlerter {
        alpha: f64,
        residuals: Vec<f64>,
    }

    impl ConformalAlerter {
        pub fn new(alpha: f64) -> Self {
            Self {
                alpha: alpha.clamp(1e-6, 0.5),
                residuals: Vec::new(),
            }
        }

        pub fn update(&mut self, observed: f64, predicted: f64) {
            self.residuals.push((observed - predicted).abs());
            if self.residuals.len() > 4096 {
                self.residuals.remove(0);
            }
        }

        pub fn threshold(&self) -> f64 {
            if self.residuals.is_empty() {
                return f64::INFINITY;
            }
            let mut values = self.residuals.clone();
            values.sort_by(|a, b| a.total_cmp(b));
            let n = values.len();
            let rank = (((1.0 - self.alpha) * (n as f64 + 1.0)).ceil() as usize).saturating_sub(1);
            values[rank.min(n - 1)]
        }

        pub fn alerts(&self, observed: f64, predicted: f64) -> bool {
            (observed - predicted).abs() > self.threshold()
        }
    }

    #[derive(Debug, Clone, Hash, PartialEq, Eq)]
    pub struct MondrianBucket {
        pub mode: String,
        pub diff: String,
        pub size: String,
    }

    #[derive(Debug, Clone)]
    pub struct MondrianRiskGate {
        alpha: f64,
        buckets: HashMap<MondrianBucket, Vec<f64>>,
        fallback_mode_diff: HashMap<(String, String), Vec<f64>>,
        fallback_mode: HashMap<String, Vec<f64>>,
        global: Vec<f64>,
    }

    impl MondrianRiskGate {
        pub fn new(alpha: f64) -> Self {
            Self {
                alpha: alpha.clamp(1e-6, 0.5),
                buckets: HashMap::new(),
                fallback_mode_diff: HashMap::new(),
                fallback_mode: HashMap::new(),
                global: Vec::new(),
            }
        }

        pub fn update(&mut self, bucket: MondrianBucket, observed: f64, predicted: f64) {
            let r = (observed - predicted).abs();
            self.buckets.entry(bucket.clone()).or_default().push(r);
            self.fallback_mode_diff
                .entry((bucket.mode.clone(), bucket.diff.clone()))
                .or_default()
                .push(r);
            self.fallback_mode
                .entry(bucket.mode.clone())
                .or_default()
                .push(r);
            self.global.push(r);
        }

        fn quantile(alpha: f64, values: &[f64]) -> Option<f64> {
            if values.is_empty() {
                return None;
            }
            let mut v = values.to_vec();
            v.sort_by(|a, b| a.total_cmp(b));
            let n = v.len();
            let rank = (((1.0 - alpha) * (n as f64 + 1.0)).ceil() as usize).saturating_sub(1);
            Some(v[rank.min(n - 1)])
        }

        pub fn risk_upper_bound(&self, bucket: &MondrianBucket, predicted: f64) -> f64 {
            let q = self
                .buckets
                .get(bucket)
                .and_then(|v| Self::quantile(self.alpha, v))
                .or_else(|| {
                    self.fallback_mode_diff
                        .get(&(bucket.mode.clone(), bucket.diff.clone()))
                        .and_then(|v| Self::quantile(self.alpha, v))
                })
                .or_else(|| {
                    self.fallback_mode
                        .get(&bucket.mode)
                        .and_then(|v| Self::quantile(self.alpha, v))
                })
                .or_else(|| Self::quantile(self.alpha, &self.global))
                .unwrap_or(0.0);
            predicted + q
        }

        pub fn is_risky(&self, bucket: &MondrianBucket, predicted: f64, budget: f64) -> bool {
            self.risk_upper_bound(bucket, predicted) > budget
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn conformal_alert_threshold_increases_with_residuals() {
            let mut c = ConformalAlerter::new(0.1);
            c.update(12.0, 10.0);
            c.update(20.0, 10.0);
            assert!(c.threshold() >= 2.0);
        }

        #[test]
        fn mondrian_fallback_works() {
            let mut gate = MondrianRiskGate::new(0.1);
            let a = MondrianBucket {
                mode: "inline".into(),
                diff: "dirty".into(),
                size: "small".into(),
            };
            gate.update(a.clone(), 12.0, 10.0);

            let unseen = MondrianBucket {
                mode: "inline".into(),
                diff: "dirty".into(),
                size: "large".into(),
            };
            let ub = gate.risk_upper_bound(&unseen, 10.0);
            assert!(ub >= 10.0);
        }
    }
}

pub mod fairness {
    use std::collections::HashMap;

    pub fn jain_index(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 1.0;
        }
        let sum: f64 = values.iter().sum();
        let sq_sum: f64 = values.iter().map(|x| x * x).sum();
        if sq_sum == 0.0 {
            return 1.0;
        }
        (sum * sum) / (values.len() as f64 * sq_sum)
    }

    #[derive(Debug, Clone)]
    pub struct InputFairnessGuard {
        counts: HashMap<String, u64>,
        latency_threshold_ms: u64,
        fairness_threshold: f64,
    }

    impl InputFairnessGuard {
        pub fn new() -> Self {
            Self {
                counts: HashMap::new(),
                latency_threshold_ms: 200,
                fairness_threshold: 0.8,
            }
        }

        pub fn observe(&mut self, source: impl Into<String>) {
            *self.counts.entry(source.into()).or_insert(0) += 1;
        }

        pub fn fairness(&self) -> f64 {
            let vals: Vec<f64> = self.counts.values().map(|v| *v as f64).collect();
            jain_index(&vals)
        }

        pub fn should_yield(&self, input_latency_ms: u64) -> bool {
            input_latency_ms > self.latency_threshold_ms
                || self.fairness() < self.fairness_threshold
        }
    }

    impl Default for InputFairnessGuard {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn jain_index_bounds_hold() {
            let v = vec![1.0, 1.0, 1.0];
            assert!((jain_index(&v) - 1.0).abs() < 1e-9);
            let w = vec![3.0, 0.0, 0.0];
            assert!((jain_index(&w) - (1.0 / 3.0)).abs() < 1e-9);
        }

        #[test]
        fn fairness_guard_detects_unfairness() {
            let mut guard = InputFairnessGuard::new();
            for _ in 0..100 {
                guard.observe("mouse");
            }
            guard.observe("key");
            assert!(guard.should_yield(10));
        }
    }
}

pub mod luminance {
    pub fn perceived_luminance(r: u8, g: u8, b: u8) -> f64 {
        0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64
    }

    pub fn is_dark_background(r: u8, g: u8, b: u8) -> bool {
        perceived_luminance(r, g, b) < 128.0
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn black_is_dark_and_white_is_not() {
            assert!(is_dark_background(0, 0, 0));
            assert!(!is_dark_background(255, 255, 255));
        }
    }
}

pub mod hover {
    #[derive(Debug, Clone)]
    pub struct CusumHoverStabilizer {
        s: f64,
        k: f64,
        h: f64,
    }

    impl CusumHoverStabilizer {
        pub fn new(k: f64, h: f64) -> Self {
            Self { s: 0.0, k, h }
        }

        pub fn update(&mut self, signed_distance: f64) -> bool {
            self.s = (self.s + signed_distance - self.k).max(0.0);
            if self.s > self.h {
                self.s = 0.0;
                true
            } else {
                false
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn small_jitter_does_not_switch() {
            let mut stab = CusumHoverStabilizer::new(0.2, 2.0);
            for _ in 0..20 {
                assert!(!stab.update(0.1));
            }
        }

        #[test]
        fn sustained_crossing_switches() {
            let mut stab = CusumHoverStabilizer::new(0.1, 1.0);
            let mut switched = false;
            for _ in 0..20 {
                if stab.update(0.3) {
                    switched = true;
                    break;
                }
            }
            assert!(switched);
        }
    }
}

pub mod pulses {
    pub fn half_sine_pulse(t: f64) -> f64 {
        let tt = t.clamp(0.0, 1.0);
        (std::f64::consts::PI * tt).sin()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn pulse_is_zero_at_bounds_and_one_midway() {
            assert!(half_sine_pulse(0.0).abs() < 1e-12);
            assert!((half_sine_pulse(0.5) - 1.0).abs() < 1e-12);
            assert!(half_sine_pulse(1.0).abs() < 1e-12);
        }
    }
}

pub mod stagger {
    use crate::animation::Easing;

    #[derive(Debug, Clone)]
    pub struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        pub fn new(seed: u64) -> Self {
            Self { state: seed.max(1) }
        }

        pub fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        pub fn next_f64(&mut self) -> f64 {
            (self.next_u64() as f64) / (u64::MAX as f64)
        }
    }

    pub fn stagger_offsets(
        n: usize,
        duration: f64,
        easing: Easing,
        jitter_max: f64,
        seed: u64,
    ) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![0.0];
        }

        let mut rng = XorShift64::new(seed);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            let base = duration * easing.apply(t);
            let jitter = ((rng.next_f64() * 2.0) - 1.0) * jitter_max;
            out.push((base + jitter).clamp(0.0, duration));
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn deterministic_with_same_seed() {
            let a = stagger_offsets(8, 100.0, Easing::EaseInOut, 3.0, 42);
            let b = stagger_offsets(8, 100.0, Easing::EaseInOut, 3.0, 42);
            assert_eq!(a, b);
        }
    }
}

pub mod height {
    #[derive(Debug, Clone)]
    pub struct BayesianHeightPredictor {
        mu0: f64,
        kappa0: f64,
        alpha: f64,
        n: usize,
        sum: f64,
        mean: f64,
        m2: f64,
        residuals: Vec<f64>,
    }

    impl BayesianHeightPredictor {
        pub fn new(mu0: f64, kappa0: f64, _sigma0: f64, alpha: f64) -> Self {
            Self {
                mu0,
                kappa0: kappa0.max(1e-9),
                alpha: alpha.clamp(1e-6, 0.5),
                n: 0,
                sum: 0.0,
                mean: 0.0,
                m2: 0.0,
                residuals: Vec::new(),
            }
        }

        pub fn observe(&mut self, h: f64) {
            let pred = self.posterior_mean();
            self.residuals.push((h - pred).abs());
            if self.residuals.len() > 4096 {
                self.residuals.remove(0);
            }

            self.n += 1;
            self.sum += h;

            if self.n == 1 {
                self.mean = h;
                self.m2 = 0.0;
            } else {
                let delta = h - self.mean;
                self.mean += delta / self.n as f64;
                let delta2 = h - self.mean;
                self.m2 += delta * delta2;
            }
        }

        pub fn posterior_mean(&self) -> f64 {
            if self.n == 0 {
                return self.mu0;
            }
            (self.kappa0 * self.mu0 + self.sum) / (self.kappa0 + self.n as f64)
        }

        pub fn sample_variance(&self) -> Option<f64> {
            if self.n < 2 {
                None
            } else {
                Some(self.m2 / (self.n as f64 - 1.0))
            }
        }

        pub fn conformal_interval(&self) -> (f64, f64) {
            let mu = self.posterior_mean();
            if self.residuals.is_empty() {
                return (mu, mu);
            }
            let mut rs = self.residuals.clone();
            rs.sort_by(|a, b| a.total_cmp(b));
            let n = rs.len();
            let rank = (((1.0 - self.alpha) * (n as f64 + 1.0)).ceil() as usize).saturating_sub(1);
            let q = rs[rank.min(n - 1)];
            (mu - q, mu + q)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn posterior_mean_moves_toward_observations() {
            let mut p = BayesianHeightPredictor::new(10.0, 5.0, 1.0, 0.1);
            let before = p.posterior_mean();
            p.observe(30.0);
            let after = p.posterior_mean();
            assert!(after > before);
            assert!(after < 30.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_modules_compile() {
        let _ = bayesian::BayesianScorer::new();
        let _ = bocpd::RegimeDetector::new();
        let _ = cusum::CusumDetector::new();
        let _ = eprocess::EProcess::new();
        let _ = fenwick::FenwickTree::new(10);
        let _ = animation::Animation::new(0.0, 1.0, std::time::Duration::from_millis(100));
        let _ = diff::DirtySpanTracker::new(80);
        let _ = hints::BayesianHintRanker::new();
        let _ = capability::CapabilityPosterior::new(0.5);
        let _ = voi::VoiSampler::new();
        let _ = conformal::ConformalAlerter::new(0.1);
        let _ = fairness::InputFairnessGuard::new();
        let _ = hover::CusumHoverStabilizer::new(0.1, 1.0);
        let _ = pulses::half_sine_pulse(0.5);
        let _ = stagger::stagger_offsets(3, 1.0, animation::Easing::Linear, 0.0, 1);
        let _ = height::BayesianHeightPredictor::new(20.0, 10.0, 1.0, 0.1);
    }

    #[test]
    fn height_predictor_updates_and_bounds() {
        let mut p = height::BayesianHeightPredictor::new(20.0, 10.0, 1.0, 0.1);
        p.observe(22.0);
        p.observe(19.0);
        p.observe(24.0);

        let mean = p.posterior_mean();
        let (lo, hi) = p.conformal_interval();
        assert!(mean >= lo && mean <= hi);
        assert!(hi >= lo);
    }

    #[test]
    fn height_predictor_uses_welford_variance() {
        let mut p = height::BayesianHeightPredictor::new(0.0, 1.0, 1.0, 0.1);
        for v in [1.0, 2.0, 3.0, 4.0] {
            p.observe(v);
        }
        assert!(p.sample_variance().unwrap_or(0.0) > 0.0);
    }
}
