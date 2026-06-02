import matplotlib.pyplot as plt

models = ["FFNN", "LSTM", "1D-CNN"]
training_energy = [6.00e-05, 1.13e-04, 4.63e-04]
f1_scores = [0.7920, 0.7730, 0.8009]

plt.figure(figsize=(7, 5))

plt.scatter(training_energy, f1_scores)

# Custom label positions so labels do not overlap or go outside the figure
label_offsets = {
    "FFNN": (8, 6),
    "LSTM": (8, 6),
    "1D-CNN": (-45, -2)
}

for model, x, y in zip(models, training_energy, f1_scores):
    plt.annotate(
        model,
        (x, y),
        textcoords="offset points",
        xytext=label_offsets[model],
        ha="left"
    )

plt.xlabel("Training energy consumption (kWh)")
plt.ylabel("F1-score")
plt.title("Trade-off between F1-score and training energy consumption")

# Adjust axis limits to make the differences clearer and keep labels inside
plt.xlim(4.0e-05, 5.0e-04)
plt.ylim(0.770, 0.805)

plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig("tradeoff_scatterplot.png", dpi=300)
plt.show()