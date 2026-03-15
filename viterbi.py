# viterbi.py

import numpy as np

def viterbi(obs_seq, states, start_prob, trans_prob, emit_prob, min_persist=5):
    """
    Implements constrained Viterbi decoding for financial regime detection.

    Parameters:
    - obs_seq: list of observation tuples, e.g., [("Low", "High", "Neutral"), ...]
    - states: list of HMM states: ["Bull", "Sideways", "Bear"]
    - start_prob: dict of initial state probabilities
    - trans_prob: dict of transition probabilities
    - emit_prob: dict of emission probabilities per state
    - min_persist: minimum number of days to persist in a state

    Returns:
    - best_path: list of inferred hidden states (market regimes)
    """
    T = len(obs_seq)
    N = len(states)

    state_to_idx = {s: i for i, s in enumerate(states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    # Dynamic programming tables
    dp = -np.inf * np.ones((T, N))
    backpointer = -1 * np.ones((T, N), dtype=int)

    # Initialization
    for s in states:
        s_idx = state_to_idx[s]
        obs = obs_seq[0]
        emis_p = emit_prob[s].get(obs, 1e-8)
        dp[0][s_idx] = np.log(start_prob[s]) + np.log(emis_p)

    # Recursion
    for t in range(1, T):
        obs = obs_seq[t]
        for curr in states:
            curr_idx = state_to_idx[curr]
            max_prob = -np.inf
            best_prev = -1

            for prev in states:
                prev_idx = state_to_idx[prev]

                # Constraint: no Bull → Bear direct transition
                if prev == "Bull" and curr == "Bear":
                    continue

                # Enforce persistence
                if t >= min_persist:
                    recent_states = backpointer[t-min_persist:t, prev_idx]
                    if not np.all(recent_states == prev_idx) and prev != curr:
                        continue

                trans_p = trans_prob[prev].get(curr, 0)
                emis_p = emit_prob[curr].get(obs, 1e-8)

                prob = dp[t-1][prev_idx] + np.log(trans_p) + np.log(emis_p)

                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_idx

            dp[t][curr_idx] = max_prob
            backpointer[t][curr_idx] = best_prev

    # Termination and backtrace
    best_path = []
    last_state = np.argmax(dp[T-1])

    for t in reversed(range(T)):
        best_path.append(idx_to_state[last_state])
        last_state = backpointer[t][last_state]

    best_path.reverse()
    return best_path

# Test Block

if __name__ == "__main__":
    # Example tiny input for sanity check
    obs_seq = [("High", "Low", "Overbought"), 
               ("Medium", "Low", "Neutral"), 
               ("Low", "High", "Oversold")]

    states = ["Bull", "Sideways", "Bear"]

    start_prob = {"Bull": 0.6, "Sideways": 0.3, "Bear": 0.1}
    
    trans_prob = {
        "Bull": {"Bull": 0.85, "Sideways": 0.15, "Bear": 0.0},
        "Sideways": {"Bull": 0.1, "Sideways": 0.8, "Bear": 0.1},
        "Bear": {"Bull": 0.05, "Sideways": 0.2, "Bear": 0.75},
    }

    emit_prob = {
        "Bull": {("High", "Low", "Overbought"): 0.4,
                 ("Medium", "Low", "Neutral"): 0.3,
                 ("Low", "High", "Oversold"): 0.1},
        "Sideways": {("High", "Low", "Overbought"): 0.2,
                     ("Medium", "Low", "Neutral"): 0.5,
                     ("Low", "High", "Oversold"): 0.2},
        "Bear": {("High", "Low", "Overbought"): 0.1,
                 ("Medium", "Low", "Neutral"): 0.2,
                 ("Low", "High", "Oversold"): 0.6},
    }

    path = viterbi(obs_seq, states, start_prob, trans_prob, emit_prob, min_persist=1)
    print("Predicted Regime Sequence:", path)
