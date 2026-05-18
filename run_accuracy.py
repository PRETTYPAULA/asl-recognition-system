from evaluate import evaluate_model


def main() -> None:
    _, accuracy_percentage = evaluate_model()

    print("=" * 66)
    print(" ASL RECOGNITION (CNN VGG16)")
    print("=" * 66)
    print("Algorithm Model           | Accuracy (%)")
    print("-" * 66)
    print(f"CNN VGG16                 | {accuracy_percentage:0.2f}%")
    print("=" * 66)


if __name__ == "__main__":
    main()
