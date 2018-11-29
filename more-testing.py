# Generate a synthetic dataset
true_w = np.array([1, 2, 3, 4, 5])
d = len(true_w)
points = []
for i in range(100000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    points.append((x, y))
