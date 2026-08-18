from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_creater import create_data_splits, validate_data_splits
from Model import MLPClassifier


condition = "(x**2 + y**2) <= 1.0**2"  # 修改后下次运行会立即重建数据
TASK_DIR = Path(__file__).resolve().parent
TRAIN_PATH = TASK_DIR / "train_data.csv"
VAL_PATH = TASK_DIR / "val_data.csv"


def main():
    # 每次都依照当前 condition 确定性重建，避免旧 CSV 静默失效。
    create_data_splits(
        train_n=800,
        val_n=200,
        train_out_path=TRAIN_PATH,
        val_out_path=VAL_PATH,
        condition=condition,
        variance=1.0,
        seed=42,
    )
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    validate_data_splits(train_df, val_df, condition)
    print(
        "data check passed: "
        f"train={len(train_df)}, val={len(val_df)}, overlap=0, "
        f"positive_ratio={train_df['label'].mean():.3f}/{val_df['label'].mean():.3f}"
    )

    model = MLPClassifier(
        train_set=train_df,
        val_set=val_df,
        Learning_rate=0.1,
        batch_size=20,
        epochs=1000,
        seed=0,
    )
    model.fit()
    val_loss, val_accuracy = model.evaluate(val_df)
    print(f"final validation: loss={val_loss:.4f}, accuracy={val_accuracy:.3f}")

    prediction = model.predict(val_df)
    plt.scatter(val_df["x"], val_df["y"], c=prediction, cmap="coolwarm", s=16)
    plt.title(f"validation predictions (accuracy={val_accuracy:.3f})")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
