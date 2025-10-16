
suppressWarnings(suppressMessages({
  library(ggplot2); library(scales); library(dplyr)
}))
age_path <- "cluster_4_0_age_mix.csv"; aud_path <- "cluster_4_0_audience.csv"
png_age <- "cluster_4_0_age_mix.png";    png_aud <- "cluster_4_0_audience.png"

# Age Mix
age_df <- read.csv(age_path, encoding="UTF-8")
age_df$bucket <- factor(age_df$bucket, levels=age_df$bucket[order(age_df$value,decreasing=TRUE)])
p1 <- ggplot(age_df, aes(x=bucket, y=value)) +
  geom_col(width=.6) +
  geom_text(aes(label=percent(value, accuracy=1)), vjust=-.3, size=5) +
  scale_y_continuous(labels=percent, limits=c(0,1)) +
  labs(title="연령 믹스(점유율, %) — Cluster 0 / k=4",
       subtitle="큰 막대일수록 손님 비중이 큼", x="연령대", y="비중(%)",
       caption="비율(0~1)을 %로 변환하여 표기") +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold", size=18))
ggsave(png_age, p1, width=8, height=5, dpi=150)

# Audience (한국어 라벨 + 강/보통/약)
aud_df <- read.csv(aud_path, encoding="UTF-8") |>
  mutate(metric=recode(metric,"NEW"="신규","REU"="단골","RES"="예약","WORK"="직장인","FLOW"="유동"),
         band=cut(value, breaks=c(-Inf,.50,.75,Inf), labels=c("약함","보통","강함"))) |>
  filter(!is.na(value), value>=0.60) |>
  arrange(desc(value))
aud_df$metric <- factor(aud_df$metric, levels=aud_df$metric)
p2 <- ggplot(aud_df, aes(x=metric, y=value, fill=band)) +
  geom_col(width=.6) +
  geom_text(aes(label=percent(value, accuracy=1)), vjust=-.3, size=5) +
  scale_y_continuous(labels=percent, limits=c(0,1)) +
  scale_fill_manual(values=c("강함"="#666666","보통"="#999999","약함"="#CCCCCC")) +
  labs(title="오디언스 신호(%) — Cluster 0 / k=4",
       subtitle="단골/유동/신규/예약/직장인 (0~1 스코어 → %)", x="오디언스", y="수준(%)", fill="강도",
       caption="임계값 미만(예: 60%)은 제외") +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold", size=18), legend.position="top")
ggsave(png_aud, p2, width=8, height=5, dpi=150)
