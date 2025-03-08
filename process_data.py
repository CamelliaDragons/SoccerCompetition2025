import pandas as pd


def goal_data(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    ゴール50フレームのデータを抽出する関数
    """
    df.loc[:, "is_play_on"] = df["playmode"] == "play_on"
    df.loc[:, "id"] = (
        df["is_play_on"] & ~df["is_play_on"].shift(1, fill_value=False)
    ).cumsum()
    df.loc[:, "is_last_play_on"] = df["playmode"].eq("play_on") & (
        df["playmode"].shift(-1) != "play_on"
    )
    df.loc[:, "goal_type"] = None
    df.loc[
        df["is_last_play_on"] & (df["playmode"].shift(-1) == "goal_l"),
        "goal_type",
    ] = "goal_l"
    df.loc[
        df["is_last_play_on"] & (df["playmode"].shift(-1) == "goal_r"),
        "goal_type",
    ] = "goal_r"
    df.loc[:, "goal_type_group"] = df.groupby("id")["goal_type"].transform(
        lambda x: x.ffill().bfill()
    )
    result = df[df["goal_type_group"].notnull()].copy()
    result = result.drop(columns=["is_play_on", "is_last_play_on", "goal_type"])
    result = result[result["playmode"] == "play_on"]
    result = result.rename(columns={"goal_type_group": "goal_type"})
    results = [group.drop(columns=["id"]) for _, group in result.groupby("id")]
    results = [df.tail(50) for df in results if len(df) > 50]

    return results


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    d = {
        "AEteam": 0,
        "CYRUS": 1,
        "FRA-UNIted": 2,
        "HELIOS2024": 3,
        "ITAndroids": 4,
        "Mars": 5,
        "Oxsy": 6,
        "R2D2": 7,
        "RoboCIn": 8,
        "YuShan2024": 9,
    }
    df["l_name"] = df["l_name"].map(d)
    df["r_name"] = df["r_name"].map(d)
    return df


def frame(df: pd.DataFrame) -> pd.DataFrame:
    df["frame"] = df.reset_index().index
    return df


def process_data(df: pd.DataFrame) -> list[pd.DataFrame]:
    dfs = goal_data(df)
    dfs = [one_hot(df) for df in dfs]
    dfs = [frame(df) for df in dfs]
    return dfs


# df = pd.read_csv(
#     "/root/robocup/robocup2d_data/aeteam2024-oxsy2024/1216-1024-aeteam2024-oxsy2024-0002-sim03.tracking.csv"
# )

# dfs = prosess(df)
# for i in dfs:
#     print(i["goal_type"].iloc[0])
# dfs[1].head()
