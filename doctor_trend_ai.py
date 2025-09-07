from dotenv import load_dotenv
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

plt.rcParams["font.family"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False #

CSV_PATH = "taipei_doctors.csv"  # 你的 CSV 檔案名稱

def parse_year(val: str) -> int:
    # "111年底" -> 2022
    if isinstance(val, str) and "底" in val:
        num = int(val.replace("年底", "").strip())
        return num + 1911
    return int(val)

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到資料檔：{path}")
    df = pd.read_csv(path)

    # ---- 欄位轉換 ----
    df["Year"] = df["統計期"].apply(parse_year).astype(int)

    # 醫師總數（西醫+中醫+牙醫）
    df["Doctors"] = df["西醫師人數"] + df["中醫師人數"] + df["牙醫師人數"]

    # 非醫師人員 = 總人數 - 醫師
    df["OtherStaff"] = df["總人數"] - df["Doctors"]

    # 全體醫事人員
    df["AllStaff"] = df["總人數"]

    # 只留近三年（依年份排序後取最後 3 筆）
    df = df.sort_values("Year").tail(3).reset_index(drop=True)

    # 年增量（針對醫師與全體）
    df["YoY_Doctors"] = df["Doctors"].diff()
    df["YoY_All"] = df["AllStaff"].diff()

    # df = df[["Year", "Doctors", "AllStaff"]].sort_values("Year").reset_index(drop=True)

    return df

def plot_trend(df: pd.DataFrame):
    years = df["Year"].astype(str)  # 用字串避免 x 軸小數點
    plt.figure(figsize=(9,5))
    plt.plot(years, df["Doctors"], marker="o", label="醫師人數")
    plt.plot(years, df["OtherStaff"], marker="s", label="非醫師人員")
    plt.plot(years, df["AllStaff"], marker="^", label="總醫事人員")
    plt.title("台北市醫師 & 醫事人員趨勢")
    plt.xlabel("Year")
    plt.ylabel("人數")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trend.png", dpi=160)
    plt.close()

def plot_yoy(df: pd.DataFrame):
    years = df["Year"].astype(str)
    plt.figure(figsize=(9,5))
    width = 0.35
    x = range(len(df))
    plt.bar([i - width/2 for i in x], df["YoY_Doctors"].fillna(0), width=width, label="醫師增量")
    plt.bar([i + width/2 for i in x], df["YoY_All"].fillna(0), width=width, label="全體增量")
    plt.xticks(ticks=x, labels=years)
    plt.title("台北市醫師 & 全體醫事人員 年增量")
    plt.xlabel("Year")
    plt.ylabel("人數變化")
    plt.legend()
    plt.tight_layout()
    plt.savefig("yoy.png", dpi=160)
    plt.close()

def plot_yearly_count(df: pd.DataFrame):
    years = df["Year"].astype(str)
    plt.figure(figsize=(9,5))
    width = 0.35
    x = range(len(df))
    plt.bar([i - width/2 for i in x], df["Doctors"], width=width, label="醫師人數")
    plt.bar([i + width/2 for i in x], df["AllStaff"], width=width, label="全體醫事人員")
    plt.xticks(ticks=x, labels=years)
    plt.title("台北市醫師 & 全體醫事人員 每年人數")
    plt.xlabel("Year")
    plt.ylabel("人數")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trend_yearly.png", dpi=160)
    plt.close()

def rule_based_summary(df: pd.DataFrame) -> str:
    y0, y1 = int(df["Year"].iloc[0]), int(df["Year"].iloc[-1])
    d0, d1 = int(df["Doctors"].iloc[0]), int(df["Doctors"].iloc[-1])
    a0, a1 = int(df["AllStaff"].iloc[0]), int(df["AllStaff"].iloc[-1])

    lines = []
    lines.append(f"期間 {y0}–{y1}：醫師人數由 {d0:,} 增至 {d1:,}；全體醫事人員由 {a0:,} 墴至 {a1:,}。")
    if df["YoY_Doctors"].notna().any():
        r = df.loc[df["YoY_Doctors"].idxmax()]
        lines.append(f"醫師單年最大增量：{int(r['Year'])} 年 +{int(r['YoY_Doctors']):,} 人。")
    if df["YoY_All"].notna().any():
        r = df.loc[df["YoY_All"].idxmax()]
        lines.append(f"全體單年最大增量：{int(r['Year'])} 年 +{int(r['YoY_All']):,} 人。")
    lines.append("建議搭配每萬人口指標，評估資源密度變化。")
    return "\n".join(lines)

def ai_summary(df: pd.DataFrame) -> str:
    prompt = (
        f"請根據以下台北市醫師與醫事人員近三年資料，產生一段摘要：\n"
        f"{df.to_string(index=False)}\n"
        "請用繁體中文簡要說明趨勢與重點。"
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # 或 "gpt-4"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def main():
    df = load_data(CSV_PATH)
    plot_trend(df)
    plot_yoy(df)
    plot_yearly_count(df)
    # 串接 AI 產生摘要
    ai_text = ai_summary(df)
    with io.open("summary.txt", "w", encoding="utf-8") as f:
        f.write(ai_text)
    print("✅ 分析完成：trend.png, yoy.png, trend_yearly.png, summary.txt 已產生")

if __name__ == "__main__":
    main()
