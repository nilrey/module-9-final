import argparse

import numpy as np
import pandas as pd


def load_events(path: str) -> pd.DataFrame:
    """Загружает события пользователей и отмечает покупки."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["is_purchase_event"] = (
        df["event"].astype(str).str.lower().isin(["purchase", "transaction"])
        | df.get("transactionid", pd.Series([None] * len(df))).notna()
    )
    return df


def load_item_properties(path: str) -> pd.DataFrame:
    """Загружает свойства товаров (только простые поля)."""
    df = pd.read_csv(path, usecols=["itemid", "property", "value"])
    df.columns = [c.lower().strip() for c in df.columns]

    # Возьмём только базовые свойства, чтобы не взорваться по памяти
    simple_props = df[df["property"].isin(["categoryid", "available"])]
    # Последние значения свойств
    simple_props = simple_props.sort_values(["itemid", "property"])
    latest = simple_props.groupby(["itemid", "property"], as_index=False).last()

    # Поворот: property -> колонка
    pivot = latest.pivot_table(
        index="itemid", columns="property", values="value", aggfunc="last"
    )
    pivot = pivot.reset_index()

    # Убедимся, что названия колонок корректные
    pivot.columns.name = None
    return pivot


def compute_item_popularity(events: pd.DataFrame) -> pd.DataFrame:
    """Считает простую популярность товаров."""
    views = (
        events[events["event"].str.lower() == "view"]
        .groupby("itemid")
        .size()
        .rename("views")
    )
    buys = (
        events[events["is_purchase_event"]].groupby("itemid").size().rename("purchases")
    )

    df = pd.concat([views, buys], axis=1).fillna(0)
    df["ctr"] = df["purchases"] / np.maximum(df["views"], 1)
    df = df.reset_index()
    return df


def build_target(
    views: pd.DataFrame, purchases: pd.DataFrame, window_hours=24
) -> pd.DataFrame:
    """Создаёт целевую переменную: купил ли пользователь товар в течение 24 часов после просмотра."""
    window = pd.Timedelta(hours=window_hours)

    purchases = purchases.rename(columns={"timestamp": "purchase_time"})
    df = views.merge(purchases, on=["visitorid", "itemid"], how="left")

    df["target"] = (
        (df["purchase_time"].notna())
        & (df["purchase_time"] > df["timestamp"])
        & (df["purchase_time"] <= df["timestamp"] + window)
    ).astype(int)

    df = df.drop(columns=["purchase_time"])
    return df


def prepare(
    events_path: str,
    item_props_path: str,
    out_item_feats_path: str,
    out_train_path: str,
    window_hours: int = 24,
):
    """Простая функция чтобы подготовить данные из одного формата в другой.
    Note:
        На входе ожидаются данные в csv формате.

    Args:
        events_path: путь до файла событий покупок
        item_props_path: путь до файла с харатеристиками товаров.
        out_item_feats_path: данные с информацией о фичах
        out_train_path: данные с фичами для тренеровки модели (таргет это вероятность покупок).
        window_hours: окно в рамках которого рассматриваем событие
    """
    events = load_events(events_path)

    views = events[events["event"].str.lower() == "view"][
        ["timestamp", "visitorid", "itemid"]
    ].copy()
    purchases = events[events["is_purchase_event"]][
        ["timestamp", "visitorid", "itemid"]
    ].copy()

    views = build_target(views, purchases, window_hours=window_hours)

    pop = compute_item_popularity(events)

    item_props = load_item_properties(item_props_path)

    item_features = pop.merge(item_props, on="itemid", how="left").fillna(0)
    views = views.merge(item_features, on="itemid", how="left").fillna(0)

    views["hour"] = views["timestamp"].dt.hour
    views["weekday"] = views["timestamp"].dt.dayofweek

    # Приведение типов, чтобы pyarrow не ругался ---
    for col in item_features.columns:
        if item_features[col].dtype == object:
            item_features[col] = item_features[col].astype(str)
    for col in views.columns:
        if views[col].dtype == object:
            views[col] = views[col].astype(str)

    item_features.to_parquet(out_item_feats_path, index=False)
    views.to_parquet(out_train_path, index=False)


if __name__ == "__main__":

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--events", default="events.csv")
        parser.add_argument("--item-props", default="item_properties_part1.csv")
        parser.add_argument("--out-item-feats", default="item_features.parquet")
        parser.add_argument("--out-train", default="data_for_training.parquet")
        parser.add_argument("--window-hours", type=int, default=24)
        parser.add_argument("--sample", type=int, default=None)
        args = parser.parse_args()

        prepare(
            args.events,
            args.item_props,
            args.out_item_feats,
            args.out_train,
            args.window_hours,
        )

    main()
