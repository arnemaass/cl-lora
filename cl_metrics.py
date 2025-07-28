import numpy as np

def forgetting_and_intransigence(acc_matrix, joint_acc=None):
    """
    acc_matrix: Lower-triangular array of shape (K, K) where a[k, j] is the accuracy on
                task j+1 after training up through task k+1 (1-based). Unfilled entries
                can be np.nan.
    joint_acc:  Length-K array where joint_acc[k] is the reference joint-training accuracy
                a*_k on task k+1. If omitted, only forgetting is returned; computing
                intransigence requires joint-training results.

    Returns:
      F_per_k:     Length-K array; entry k is the mean forgetting F_{k+1} (0 when k=0).
      I_per_k:     Length-K array; entry k is the intransigence I_{k+1}
                   (np.nan if joint_acc is None).
      f_taskwise:  (K, K) matrix of f^{(k)}_j (defined for j<k; others are np.nan).
    """
    K = acc_matrix.shape[0]
    f_taskwise = np.full((K, K), np.nan, dtype=float)
    F_per_k = np.zeros(K, dtype=float)
    I_per_k = np.full(K, np.nan if joint_acc is None else 0.0, dtype=float)

    # Compute task-level forgetting
    for k in range(1, K):  # k denotes "after finishing task k+1"
        for j in range(0, k):  # only for past tasks j < k
            hist_best = np.nanmax(acc_matrix[:k, j])  # best historical performance up to step k-1
            now = acc_matrix[k, j]
            f = hist_best - now
            f_taskwise[k, j] = f
        F_per_k[k] = np.nanmean(f_taskwise[k, :k]) if k > 0 else 0.0

    # Compute intransigence
    if joint_acc is not None:
        joint_acc = np.asarray(joint_acc, dtype=float)
        for k in range(K):
            if not np.isnan(acc_matrix[k, k]) and not np.isnan(joint_acc[k]):
                I_per_k[k] = joint_acc[k] - acc_matrix[k, k]

    return F_per_k, I_per_k, f_taskwise
