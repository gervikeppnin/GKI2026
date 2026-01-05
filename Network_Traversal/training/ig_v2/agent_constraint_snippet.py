    def _check_monotonicity_penalty(self, theta: Dict[str, float]) -> float:
        """
        Calculates penalty if age monotonicity is violated.
        Constraint: Older pipes (lower C) <= Younger pipes (higher C).
        buckets are sorted by year (1970, 1975, ...)
        Therefore: mean(bucket[i]) <= mean(bucket[i+1])
        """
        if not self.pipe_5yr_buckets:
            return 0.0
            
        # 1. Map theta to individual pipes
        # This allows the constraint to work regardless of current grouping strategy
        pipe_roughness = {}
        for pipe, group in self.group_mapping.items():
            pipe_roughness[pipe] = theta.get(group, 100.0)
            
        # 2. Compute mean per bucket
        bucket_sums = {b: 0.0 for b in self.sorted_buckets}
        bucket_counts = {b: 0 for b in self.sorted_buckets}
        
        for pipe, r in pipe_roughness.items():
            b = self.pipe_5yr_buckets.get(pipe)
            if b is not None and b in bucket_sums:
                bucket_sums[b] += r
                bucket_counts[b] += 1
                
        # 3. Check Monotonicity
        penalty = 0.0
        prev_mean = -float('inf')
        
        for b in self.sorted_buckets:
            if bucket_counts[b] == 0:
                continue
                
            curr_mean = bucket_sums[b] / bucket_counts[b]
            
            # Constraint: prev_mean (Older) <= curr_mean (Younger)
            # Violation: curr_mean < prev_mean
            if curr_mean < prev_mean:
                diff = prev_mean - curr_mean
                # Quadratic penalty? Or linear? Linear is simpler.
                penalty += diff * 5.0 
            
            prev_mean = curr_mean
            
        return penalty
