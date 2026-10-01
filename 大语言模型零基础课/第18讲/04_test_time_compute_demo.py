import numpy as np

rng = np.random.default_rng(42)
num_tasks = 20000
# Each task has its own latent difficulty. Samples on the same task are therefore correlated.
# This is closer to reality than pretending every try has the exact same independent success rate.
p_task = rng.beta(5.0, 3.5, size=num_tasks)  # average single-sample success around 0.59

Ns = [1, 2, 4, 8, 16, 32]
print("N | random-pick | oracle-best-of-N | noisy-verifier")
print("--+-------------+------------------+---------------")
for N in Ns:
    success = rng.random((num_tasks, N)) < p_task[:, None]

    # Baseline: spend N samples but pick one at random -> essentially no benefit.
    random_pick = success[:, 0].mean()

    # Oracle verifier: if any candidate is correct, it always selects a correct one.
    oracle = success.any(axis=1).mean()

    # Noisy verifier: correct candidates tend to score higher, but not perfectly.
    score = rng.normal(0.0, 1.0, size=(num_tasks, N)) + success * 1.2
    chosen_idx = score.argmax(axis=1)
    noisy = success[np.arange(num_tasks), chosen_idx].mean()

    print(f"{N:2d} | {random_pick:11.4f} | {oracle:16.4f} | {noisy:13.4f}")

p = p_task.mean()
print("\nMean single-sample success p =", round(float(p), 4))
print("If samples were truly independent with constant p,")
print("P(at least one success) = 1 - (1-p)^N")
for N in Ns:
    indep = 1 - (1-p)**N
    print(f"N={N:2d}: independent-theory={indep:.4f}")

print("\nKey lesson:")
print("More samples only help if you can exploit diversity and select/combine candidates well.")
print("Correlated errors and imperfect verifiers reduce the gain from test-time compute.")
